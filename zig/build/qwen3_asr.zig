const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_asr/qwen3_asr.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
}

pub fn probe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_asr/check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
    const exe = b.addExecutable(.{ .name = "tf-qwen3-asr-check", .root_module = m });
    b.step("tf-qwen3-asr-check", "Export native ASR stage rows and teacher-forced logits").dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } }).step);
    const bench = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_asr/gemm_bench.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const bench_exe = b.addExecutable(.{ .name = "tf-qwen3-asr-gemm-bench", .root_module = bench });
    b.step("tf-qwen3-asr-gemm-bench", "Measure actual ASR projection weights and cuBLASLt candidates").dependOn(&b.addInstallArtifact(bench_exe, .{}).step);
    const contract = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/asr_contract.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const test_exe = b.addExecutable(.{ .name = "tf-qwen3-asr-contract", .root_module = contract });
    b.step("tf-qwen3-asr-contract", "Build focused ASR kernel diagnostics").dependOn(&b.addInstallArtifact(test_exe, .{}).step);
}
