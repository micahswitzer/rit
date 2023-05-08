const std = @import("std");
const vm = @import("riscv/vm.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    if (args.len < 2)
        return error.InvalidUsage;

    const file = try std.fs.cwd().openFileZ(args[1], .{});
    defer file.close();

    const origin = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 0) else 0;

    try vm.dumpDisassembled(file.reader(), origin);

    const contents = try alloc.alloc(u8, 4096 * 16);
    defer alloc.free(contents);

    @memset(contents, 0);

    try file.seekTo(0);
    _ = try file.readAll(contents[origin..]);

    var rvm = vm.VM.init(contents, 0, origin);

    while (true) {
        rvm.step() catch |e| {
            if (e == error.StopEmulation) {
                std.debug.print("emulation stopped due to ebreak\n", .{});
            } else {
                std.debug.print("emulation stopped due to error: {s}\n", .{@errorName(e)});
            }
            break;
        };
    }

    rvm.regs.dump();
}

test {
    _ = @import("riscv/vm.zig");
}
