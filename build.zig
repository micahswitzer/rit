const std = @import("std");

pub fn build(b: *std.Build) void {
    const riscv_target = std.zig.CrossTarget{
        .cpu_arch = .riscv32,
        .cpu_model = .{
            .explicit = &std.Target.riscv.cpu.generic_rv32,
        },
        .os_tag = .freestanding,
        .ofmt = .elf,
    };
    const riscv_optimize: std.builtin.Mode = .ReleaseSmall;

    const riscv_elf = b.addExecutable(.{
        .name = "sample",
        .root_source_file = .{ .path = "src/riscv/sample.S" },
        .target = riscv_target,
        .optimize = riscv_optimize,
        .linkage = .static,
    });
    riscv_elf.addCSourceFile("src/riscv/sample.c", &[_][]const u8{
        "-ffreestanding",
        "-nostdinc",
        "-nostartfiles",
    });
    riscv_elf.setLinkerScriptPath(.{
        .path = "src/riscv/linker.ld",
    });
    riscv_elf.strip = true;

    const riscv_bin = b.addObjCopy(riscv_elf.getOutputSource(), .{ .format = .bin });

    b.getInstallStep().dependOn(&b.addInstallBinFile(riscv_elf.getOutputSource(), "sample.elf").step);

    b.getInstallStep().dependOn(&b.addInstallBinFile(riscv_bin.getOutputSource(), "sample.bin").step);

    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "rit",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a RunStep in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    run_cmd.addFileSourceArg(riscv_bin.getOutputSource());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
