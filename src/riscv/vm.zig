const std = @import("std");
const ZigType = std.builtin.Type;

const IALIGN = 32;
const XLEN = 32;
const XSz = std.meta.Int(.unsigned, XLEN);
const gp_reg_count = 32;
const Reg = u5;

const RegisterFile = struct {
    const RA = 1;
    const SP = 2;
    const LINK = 5;

    x: [gp_reg_count]XSz,
    pc: XSz,
};

// zig fmt: off
const GPR = enum(u5) {
    // the first 16 registers are common across every RV32 ISA
    // x0
    zero = 0, 
    // x1-x2
    ra, sp, gp, tp,
    // x5-x7
    t0, t1, t2,
    // x8-x9
    fp, s1,
    // x10-x15
    a0,  a1,  a2,  a3,  a4,  a5,
    // the latter 16 registers are only present on the "full" RV32 ISA
    a6, a7,
    s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
    t3, t4, t5, t6,

    pub fn get(val: u5) @This() {
        return @intToEnum(@This(), val);
    }

    pub fn str(self: @This()) []const u8 {
        return @tagName(self);
    }
};
// zig fmt: on

const Encoding = packed union {
    const Kind = enum {
        R,
        I,
        S,
        B,
        U,
        J,
    };

    const R = packed struct(u32) {
        opcode: u7,
        rd: u5,
        funct3: u3,
        rs1: u5,
        rs2: u5,
        funct7: u7,
    };
    const I = packed struct(u32) {
        opcode: u7,
        rd: u5,
        funct3: u3,
        rs1: u5,
        imm11_0: i12,

        pub inline fn imm(self: @This()) i12 {
            return self.imm11_0;
        }
    };
    const S = packed struct(u32) {
        opcode: u7,
        imm4_0: u5,
        funct3: u3,
        rs1: u5,
        rs2: u5,
        imm11_5: u7,

        const Imm = packed struct(u12) {
            imm4_0: u5,
            imm11_5: u7,
        };

        pub inline fn imm(self: @This()) i12 {
            return @bitCast(i12, Imm{ .imm4_0 = self.imm4_0, .imm11_5 = self.imm11_5 });
        }
    };
    const B = packed struct(u32) {
        opcode: u7,
        imm11: u1,
        imm4_1: u4,
        funct3: u3,
        rs1: u5,
        rs2: u5,
        imm10_5: u6,
        imm12: u1,

        const Imm = packed struct(u13) {
            _zero: u1 = 0,
            imm4_1: u4,
            imm10_5: u6,
            imm11: u1,
            imm12: u1,
        };

        pub inline fn imm(self: @This()) i13 {
            return @bitCast(i13, Imm{ .imm4_1 = self.imm4_1, .imm10_5 = self.imm10_5, .imm11 = self.imm11, .imm12 = self.imm12 });
        }
    };
    const U = packed struct(u32) {
        opcode: u7,
        rd: u5,
        imm31_12: u20,

        const Imm = packed struct(u32) {
            _zero: u12 = 0,
            imm31_12: u20,
        };

        pub inline fn imm(self: @This()) i32 {
            // Note that this immediate can be decoded in a much more naive way
            // by simply masking off the lower bits.
            // I've decided to stick with the "regular" pattern in hopes that it
            // will compile down to the same code and potentially enable future
            // source code simplifications.
            return @bitCast(i32, Imm{ .imm31_12 = self.imm31_12 });
        }
    };
    const J = packed struct(u32) {
        opcode: u7,
        rd: u5,
        imm19_12: u8,
        imm11: u1,
        imm10_1: u10,
        imm20: u1,

        const Imm = packed struct(u21) {
            _zero: u1 = 0,
            imm10_1: u10,
            imm11: u1,
            imm19_12: u8,
            imm20: u1,
        };

        pub inline fn imm(self: @This()) i21 {
            return @bitCast(i21, Imm{ .imm10_1 = self.imm10_1, .imm11 = self.imm11, .imm19_12 = self.imm19_12, .imm20 = self.imm20 });
        }
    };
    const Op = packed struct(u32) {
        inst1_0: u2,
        inst6_2: u5,
        _ignored: u25,
    };
    /// Special version of the I encoding for fences
    const Fence = packed struct(u32) {
        const Order = packed struct(u4) {
            write: u1 = 0,
            read: u1 = 0,
            output: u1 = 0,
            input: u1 = 0,

            const RW = @This(){ .read = 1, .write = 1 };

            fn eql(self: @This(), other: @This()) bool {
                return @bitCast(u4, self) == @bitCast(u4, other);
            }

            fn str(self: @This(), buf: *[4]u8) []const u8 {
                var at: usize = 0;
                if (self.input == 1) {
                    buf[0] = 'I';
                    at = 1;
                }
                if (self.output == 1) {
                    buf[at] = 'O';
                    at += 1;
                }
                if (self.read == 1) {
                    buf[at] = 'R';
                    at += 1;
                }
                if (self.write == 1) {
                    buf[at] = 'W';
                    at += 1;
                }
                return buf[0..at];
            }
        };

        const FM = enum(u4) {
            normal = 0b0000,
            tso = 0b1000, // only defined for rw, rw; reserved otherwise
            _, // all other values are reserved for future use
        };

        opcode: u7,
        rd: u5,
        funct3: u3,
        rs1: u5,
        successor: Order,
        predecessor: Order,
        fm: FM,
    };

    raw: u32,
    op: Op,
    i_fence: Fence,
    r: R,
    i: I,
    s: S,
    b: B,
    u: U,
    j: J,

    pub inline fn from(raw: u32) @This() {
        // this function allows for potentially storing each decoding separately in the future
        // assert that this is a 32-bit instruction
        std.debug.assert(raw & 0b11 == 0b11);
        return .{ .raw = raw };
    }

    pub inline fn opcode(self: @This()) u7 {
        // it doesn't really matter where this comes from, this function is mainly just a shortcut
        return self.r.opcode;
    }
};

const Opcode = enum(u5) {
    Load = 0b00_000,
    LoadFp,
    Custom0,
    MiscMem,
    OpImm,
    Auipc,
    OpImm32,
    @"48b1",
    Store = 0b01_000,
    StoreFp,
    Custom1,
    Amo,
    Op,
    Lui,
    Op32,
    @"64b",
    MAdd = 0b10_000,
    MSub,
    NMSum,
    NMAdd,
    OpFp,
    Reserved10101,
    Custom2,
    @"48b2",
    Branch = 0b11_000,
    Jalr,
    Reserved11010,
    Jal,
    System,
    Reserved11101,
    Custom3,
    @">=80b",

    pub fn decode(insn: Encoding) @This() {
        // validate that the encoding is as we expected
        std.debug.assert(insn.op.inst1_0 == 0b11);
        // std.debug.assert(insn.op.inst6_2 & 0b111 != 0b111);
        return @intToEnum(@This(), insn.op.inst6_2);
    }
};

const Mnemonic = enum(u6) {
    Lui,
    Auipc,
    Jal,
    Jalr,
    Beq,
    Bne,
    Blt,
    Bge,
    Bltu,
    Bgeu,
    Lb,
    Lh,
    Lw,
    Lbu,
    Lhu,
    Sb,
    Sh,
    Sw,
    Addi,
    Slti,
    Sltiu,
    Xori,
    Ori,
    Andi,
    Slli,
    Srli,
    Srai,
    Add,
    Sub,
    Sll,
    Slt,
    Sltu,
    Xor,
    Srl,
    Sra,
    Or,
    And,
    Fence,
    @"Fence.Tso",
    Ecall,
    Ebreak,
    // the following are pseudo-instructions (special-cases of regular instructions)
    Nop, // ADDI x0, x0,   0  ; add 0 to 0 and discard the result
    Mv, //  ADDI rd, rs1,  0  ; add zero to rs1 and store the result in rd
    Not, // XORI rd, rs1, -1  ; xor rs1 with all ones and store in rd
    J, //   JAL  x0, addr     ; jump to addr, discarding the return address
    Jr, //  JALR x0, off(rs)  ;
    Ret, // JALR x0, 0(ra)    ;
};

const DecodeError = error{Unsupported};

// zig fmt: off
fn anotherDecode(enc: Encoding) DecodeError!Mnemonic {
    return switch (Opcode.decode(enc)) {
        .Load => switch (enc.i.funct3) {
            0b000 => .Lb,    // RV32I Base
            0b001 => .Lh,    // RV32I Base
            0b010 => .Lw,    // RV32I Base
            0b011 => error.Unsupported,
            0b100 => .Lbu,   // RV32I Base
            0b101 => .Lhu,   // RV32I Base
            0b110, 0b111 => error.Unsupported,
        },
        .LoadFp => error.Unsupported,
        .MiscMem => switch (enc.i.funct3) {
            0b000 => switch (enc.i_fence.fm) {
                .normal => .Fence,
                .tso => if (enc.i_fence.predecessor.eql(Encoding.Fence.Order.RW)
                        and enc.i_fence.successor.eql(Encoding.Fence.Order.RW))
                    .@"Fence.Tso" // only defined for FENCE.TSO RW, RW
                else
                    error.Unsupported,
                _ => error.Unsupported,
            },
            else => error.Unsupported,
        },
        .OpImm => switch (enc.i.funct3) {
            // regular I-type
            0b000 => if (enc.i.imm11_0 == 0)
                if (enc.i.rs1 == 0 and enc.i.rd == 0)
                    .Nop     // ADDI x0, x0, 0
                else
                    .Mv      // ADDI rd, rs1, 0
            else
                .Addi,       // RV32I Base
            0b010 => .Slti,  // RV32I Base
            0b011 => .Sltiu, // RV32I Base
            0b100 => if (enc.i.imm() == -1)
                .Not    // XORI rd, rs1, -1
            else
                .Xori,  // RV32I Base
            0b110 => .Ori,   // RV32I Base
            0b111 => .Andi,  // RV32I Base
            // "specialization of the I-type format"
            0b001 => switch (enc.r.funct7) {
                0b0000000 => .Slli,  // RV32I Base
                else => error.Unsupported,
            },
            0b101 => switch (enc.r.funct7) { 
                0b0000000 => .Srli,  // RV32I Base
                0b0100000 => .Srai,  // RV32I Base
                else => error.Unsupported,
            },
        },
        .Auipc => .Auipc,    // RV32I Base
        .OpImm32 => error.Unsupported,
        .Store => switch (enc.s.funct3) {
            0b000 => .Sb,    // RV32I Base
            0b001 => .Sh,    // RV32I Base
            0b010 => .Sw,    // RV32I Base
            0b011...0b111 => error.Unsupported,
        },
        .StoreFp => error.Unsupported,
        .Amo => error.Unsupported,
        .Op => switch (enc.r.funct3) {
            0b000 => switch (enc.r.  funct7) {
                0b0000000 => .Add,   // RV32I Base
                0b0100000 => .Sub,   // RV32I Base
                else => error.Unsupported,
            },
            0b001 => switch (enc.r.funct7) {
                0b0000000 => .Sll,   // RV32I Base
                else => error.Unsupported,
            },
            0b010 => switch (enc.r.funct7) {
                0b0000000 => .Slt,   // RV32I Base
                else => error.Unsupported,
            },
            0b011 => switch (enc.r.funct7) {
                0b0000000 => .Sltu,  // RV32I Base
                else => error.Unsupported,
            },
            0b100 => switch (enc.r.funct7) {
                0b0000000 => .Xor,   // RV32I Base
                else => error.Unsupported,
            },
            0b101 => switch (enc.r.funct7) {
                0b0000000 => .Srl,   // RV32I Base
                0b0100000 => .Sra,   // RV32I Base
                else => error.Unsupported,
            },
            0b110 => switch (enc.r.funct7) {
                0b0000000 => .Or,    // RV32I Base
                else => error.Unsupported,
            },
            0b111 => switch (enc.r.funct7) {
                0b0000000 => .And,   // RV32I Base
                else => error.Unsupported,
            },
        },
        .Lui => .Lui,        // RV32I Base
        .Op32 => error.Unsupported,
        .MAdd => error.Unsupported,
        .MSub => error.Unsupported,
        .NMSum => error.Unsupported,
        .NMAdd => error.Unsupported,
        .OpFp => error.Unsupported,
        .Branch => switch (enc.b.funct3) {
            0b000 => .Beq,   // RV32I Base
            0b001 => .Bne,   // RV32I Base
            0b010, 0b011 => error.Unsupported,
            0b100 => .Blt,   // RV32I Base
            0b101 => .Bge,   // RV32I Base
            0b110 => .Bltu,  // RV32I Base
            0b111 => .Bgeu,  // RV32I Base
        },
        .Jalr => switch (enc.i.funct3) {
            0b000 => if (enc.i.rd == 0)
                if (enc.i.rs1 == 1)
                    .Ret
                else
                    .Jr
            else
                .Jalr,  // RV32I Base
            else => error.Unsupported,
        },
        .Jal => if (enc.j.rd == 0)
            // pseudo-instruction
            .J               // RV32I Base
        else
            .Jal,            // RV32I Base
        .System => switch (enc.raw) {
            0b000000000000_00000_000_00000_1110011 => .Ecall,
            0b000000000001_00000_000_00000_1110011 => .Ebreak,
            else => error.Unsupported,
        },
        else => error.Unsupported,
    };
}
// zig fmt: on

const fmtBuf = std.fmt.bufPrint;

const BufArrayError = error{BufferFull};

/// Baby's first bump-allocator
fn BufArray(comptime T: type, comptime max_elements: comptime_int) type {
    const ElementCount = std.math.IntFittingRange(0, max_elements);
    return struct {
        const MAX_ELEMENTS = max_elements;

        /// The whole buffer allocated to this array
        backing: []T,
        /// How much of the buffer is used
        used: usize = 0,
        /// How much of the buffer is reserved
        reserved: usize = 0,
        /// The number of elements currently allocated
        elem_num: ElementCount = 0,
        /// Storage for the elements themselves
        elem_buf: [MAX_ELEMENTS][]T = undefined,

        fn storeAssumeCapacity(self: *@This(), item: []T) void {
            self.elem_buf[self.elem_num] = item;
            self.elem_num += 1;
            self.used += item.len;
        }

        fn reserve(self: *@This(), count: usize) BufArrayError![]T {
            std.debug.assert(self.reserved == 0);
            const next_reserved = self.used + count;
            if (next_reserved > self.backing.len)
                return error.BufferFull;
            self.reserved = next_reserved;
            return self.backing[self.used..self.reserved];
        }

        fn reserveAll(self: *@This()) []T {
            std.debug.assert(self.reserved == 0);
            self.reserved = self.backing.len;
            return self.backing[self.used..];
        }

        fn commitSlice(self: *@This(), elem: []T) void {
            // const resv_begin = self.backing[self.used..].ptr;
            // const resv_end = self.backing[self.reserved..].ptr;
            // const elem_begin = elem.ptr;
            // const elem_end = elem[elem.len..].ptr;
            // std.debug.assert(elem_begin == resv_begin and elem_end <= resv_end);
            self.storeAssumeCapacity(elem);
            self.reserved = 0;
        }

        fn storeFrom(self: *@This(), src: []const T) BufArrayError!void {
            std.debug.assert(src.len <= self.available());
            const buf = try self.reserve(src.len);
            std.mem.copyForwards(T, buf, src);
            self.commitSlice(buf);
        }

        fn available(self: @This()) usize {
            return self.backing.len - self.used;
        }

        pub fn get(self: *const @This()) []const []T {
            return self.elem_buf[0..self.elem_num];
        }
    };
}

// a mnemonic plus up to three operands
const Disassembled = BufArray(u8, 4);

pub fn Disassembler(comptime Reader: type) type {
    return struct {
        pc: u32,
        reader: Reader,

        const Error = DecodeError || Reader.Error || BufArrayError;

        /// Return value is only valid
        pub fn next(self: *@This(), buf: []u8) Error!?Disassembled {
            const enc = Encoding.from(self.reader.readIntLittle(u32) catch return null);
            const mnem = try anotherDecode(enc);
            var parts = Disassembled{ .backing = buf };
            const mnem_str = @tagName(mnem);
            const mnem_store = try parts.reserve(mnem_str.len);
            _ = std.ascii.lowerString(mnem_store, mnem_str);
            parts.commitSlice(mnem_store);
            const rd = GPR.get(enc.r.rd).str();
            const rs1 = GPR.get(enc.r.rs1).str();
            const rs2 = GPR.get(enc.r.rs2).str();
            const pc = self.pc;
            self.pc += 4;
            const kind: Encoding.Kind = switch (mnem) {
                // RV32I regular instructions
                .Lui, .Auipc => .U,
                .Jal => .J,
                .Jr, .Jalr => .I,
                .Beq, .Bne, .Blt, .Bge, .Bltu, .Bgeu => .B,
                .Lb, .Lh, .Lw, .Lbu, .Lhu => .I,
                .Sb, .Sh, .Sw => .S,
                // register-immediate
                .Addi, .Slti, .Sltiu, .Xori, .Ori, .Andi => .I,
                .Slli, .Srli, .Srai => {
                    try parts.storeFrom(rd);
                    try parts.storeFrom(rs1);
                    var tmp = parts.reserveAll();
                    // can fail...
                    // using an S-type here is a hack
                    const shamt = std.fmt.bufPrintIntToSlice(tmp, enc.s.rs2, 10, .lower, .{});
                    parts.commitSlice(shamt);
                    return parts;
                },
                // register-register
                .Add, .Sub, .Sll, .Slt, .Sltu, .Xor, .Srl, .Sra, .Or, .And => .R,
                .Fence, .@"Fence.Tso" => {
                    var tmp: [4]u8 = undefined;
                    try parts.storeFrom(enc.i_fence.predecessor.str(&tmp));
                    try parts.storeFrom(enc.i_fence.successor.str(&tmp));
                    return parts;
                },
                // no arguments
                .Ecall, .Ebreak, .Nop, .Ret => return parts,
                .Not, .Mv => {
                    try parts.storeFrom(rd);
                    try parts.storeFrom(rs1);
                    return parts;
                },
                .J => {
                    try parts.storeFrom(rd);
                    var tmp = parts.reserveAll();
                    // can fail...
                    const shamt = std.fmt.bufPrintIntToSlice(tmp, enc.j.imm(), 16, .lower, .{});
                    parts.commitSlice(shamt);
                    return parts;
                },
            };
            switch (kind) {
                .R => {
                    try parts.storeFrom(rd);
                    try parts.storeFrom(rs1);
                    try parts.storeFrom(rs2);
                },
                .I => switch (mnem) {
                    .Jr, .Jalr, .Lb, .Lh, .Lw, .Lbu, .Lhu => {
                        if (mnem != .Jr and (mnem != .Jalr or enc.i.rd != 1))
                            try parts.storeFrom(rd);
                        var tmp = parts.reserveAll();
                        const imm = (if (mnem == .Jalr and enc.i.imm() == 0)
                            std.fmt.bufPrint(tmp, "{s}", .{rs1})
                        else
                            std.fmt.bufPrint(tmp, "{}({s})", .{ enc.i.imm(), rs1 })) catch return error.BufferFull;
                        parts.commitSlice(imm);
                    },
                    .Addi, .Slti, .Sltiu, .Xori, .Ori, .Andi => {
                        try parts.storeFrom(rd);
                        try parts.storeFrom(rs1);
                        var tmp = parts.reserveAll();
                        // can fail...
                        const imm = std.fmt.bufPrintIntToSlice(tmp, enc.i.imm(), 10, .lower, .{});
                        parts.commitSlice(imm);
                    },
                    else => unreachable,
                },
                .S => {
                    try parts.storeFrom(rs2);
                    var tmp = parts.reserveAll();
                    const imm = std.fmt.bufPrint(tmp, "{}({s})", .{ enc.s.imm(), rs1 }) catch return error.BufferFull;
                    parts.commitSlice(imm);
                },
                .B => {
                    const ea = pc +% immU(enc.b.imm());
                    try parts.storeFrom(rs1);
                    try parts.storeFrom(rs2);
                    var tmp = parts.reserveAll();
                    // can fail...
                    const shamt = std.fmt.bufPrintIntToSlice(tmp, ea, 16, .lower, .{});
                    parts.commitSlice(shamt);
                },
                .U => {
                    const ea = switch (mnem) {
                        .Lui => immU(enc.u.imm()),
                        .Auipc => pc +% immU(enc.u.imm()),
                        else => unreachable,
                    };
                    try parts.storeFrom(rd);
                    var tmp = parts.reserveAll();
                    // can fail...
                    const imm = std.fmt.bufPrintIntToSlice(tmp, ea, switch (mnem) {
                        .Lui => 10,
                        .Auipc => 16,
                        else => unreachable,
                    }, .lower, .{});
                    parts.commitSlice(imm);
                },
                .J => {
                    const ea = pc +% immU(enc.j.imm());
                    try parts.storeFrom(rd);
                    var tmp = parts.reserveAll();
                    // can fail...
                    const shamt = std.fmt.bufPrintIntToSlice(tmp, ea, 16, .lower, .{});
                    parts.commitSlice(shamt);
                },
            }
            return parts;
        }
    };
}

pub fn disassembler(reader: anytype, pc: u32) Disassembler(@TypeOf(reader)) {
    return .{ .pc = pc, .reader = reader };
}

const ORG = 0x1000;

fn immS(imm: anytype) i32 {
    return @as(i32, imm);
}

fn immU(imm: anytype) u32 {
    return @bitCast(u32, immS(imm));
}

pub fn dumpDisassembled(reader: anytype, origin: u32) !void {
    var disas = disassembler(reader, origin);
    var buf: [128]u8 = undefined;
    while (try disas.next(&buf)) |res| {
        var parts = res.get();
        std.debug.print("{x:0>8}  {s: <5}", .{ disas.pc - 4, parts[0] });
        if (parts.len > 1)
            std.debug.print(" {s}", .{parts[1]});
        if (parts.len >= 2) for (parts[2..]) |part| {
            std.debug.print(", {s}", .{part});
        };
        std.debug.print("\n", .{});
    }
}

test "disassemble" {
    //const program = std.mem.sliceAsBytes(&PROGRAM);
    const program = @embedFile("sample.bin");
    var stream = std.io.fixedBufferStream(program);
    var reader = stream.reader();
    std.debug.print("\n", .{});
    try dumpDisassembled(reader, ORG);
}