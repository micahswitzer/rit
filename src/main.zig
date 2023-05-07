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
}

test {
    _ = @import("riscv/vm.zig");
}
