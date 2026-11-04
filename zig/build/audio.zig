const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const m = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libc = true });
    m.addCSourceFile(.{ .file = b.path("zig/src/audio/tf_audio.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
    for ([_][]const u8{ "avformat", "avcodec", "swresample", "avutil" }) |name| m.linkSystemLibrary(name, .{ .use_pkg_config = .force });
    const lib = b.addLibrary(.{ .name = "tfaudio", .linkage = .dynamic, .root_module = m });
    b.step("audio", "Native audio decoding over the system FFmpeg development libraries").dependOn(&b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = .{ .custom = "native/lib" } } }).step);
}
