const R = @import("registry.zig").CudaRecord;

pub const cuda = [_]R{
    .{ .launch = .{ .name = "Qwen3-ASR im2col", .symbol = "tf_asr_im2col", .image = "qwen3_asr", .threads = 256 },
        .usage = .{ .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 32, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "Qwen3-ASR convolution pack", .symbol = "tf_asr_conv_pack", .image = "qwen3_asr", .threads = 256 },
        .usage = .{ .{ .regs = 21, .static = 0, .bound = 0 }, .{ .regs = 21, .static = 0, .bound = 0 }, .{ .regs = 21, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "Qwen3-ASR audio attention", .symbol = "tf_asr_audio_attention", .image = "qwen3_asr", .threads = 128 },
        .usage = .{ .{ .regs = 31, .static = 1024, .bound = 0 }, .{ .regs = 39, .static = 1024, .bound = 0 }, .{ .regs = 39, .static = 1024, .bound = 0 }, .{ .regs = 40, .static = 1024, .bound = 0 }, .{ .regs = 40, .static = 1024, .bound = 0 } } },
};
