const std = @import("std");

pub const Config = struct {
    hidden: usize = 2048,
    intermediate: usize = 6144,
    layers: usize = 28,
    heads: usize = 16,
    kv_heads: usize = 8,
    head_dim: usize = 128,
    vocab: usize = 151936,
    context: usize = 65536,
    audio_dim: usize = 1024,
    audio_layers: usize = 24,
    audio_heads: usize = 16,
    audio_ffn: usize = 4096,
    conv_channels: usize = 480,
    eps: f32 = 1e-6,
    theta: f32 = 1000000,
    audio_token: u32 = 151676,

    pub fn parse(a: std.mem.Allocator, bytes: []const u8) !Config {
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
        if (root != .object) return error.BadConfig;
        const mt = root.object.get("model_type") orelse return error.BadConfig;
        if (mt != .string or !std.mem.eql(u8, mt.string, "qwen3_asr")) return error.BadConfig;
        const thinker = root.object.get("thinker_config") orelse return error.UnsupportedCheckpointSchema;
        const t = object(thinker, "text_config") orelse return error.BadConfig;
        const e = object(thinker, "audio_config") orelse return error.BadConfig;
        const c: Config = .{
            .hidden = try uint(t, "hidden_size"),
            .intermediate = try uint(t, "intermediate_size"),
            .layers = try uint(t, "num_hidden_layers"),
            .heads = try uint(t, "num_attention_heads"),
            .kv_heads = try uint(t, "num_key_value_heads"),
            .head_dim = try uint(t, "head_dim"),
            .vocab = try uint(t, "vocab_size"),
            .context = try uint(t, "max_position_embeddings"),
            .audio_dim = try uint(e, "d_model"),
            .audio_layers = try uint(e, "encoder_layers"),
            .audio_heads = try uint(e, "encoder_attention_heads"),
            .audio_ffn = try uint(e, "encoder_ffn_dim"),
            .conv_channels = try uint(e, "downsample_hidden_size"),
        };
        if (c.hidden != 2048 or c.intermediate != 6144 or c.layers != 28 or c.heads != 16 or
            c.kv_heads != 8 or c.head_dim != 128 or c.audio_dim != 1024 or c.audio_layers != 24 or
            c.audio_heads != 16 or c.audio_ffn != 4096 or c.conv_channels != 480 or
            try uint(e, "num_mel_bins") != 128 or try uint(e, "n_window") != 50 or
            try uint(e, "n_window_infer") != 800 or try uint(e, "output_dim") != c.hidden or
            try number(t, "rope_theta") != c.theta or try number(t, "rms_norm_eps") != c.eps)
            return error.UnsupportedQwen3AsrGeometry;
        return c;
    }
};

fn object(v: std.json.Value, key: []const u8) ?std.json.Value {
    if (v != .object) return null;
    const x = v.object.get(key) orelse return null;
    return if (x == .object) x else null;
}

fn uint(v: std.json.Value, key: []const u8) !usize {
    const x = v.object.get(key) orelse return error.BadConfig;
    if (x != .integer or x.integer <= 0) return error.BadConfig;
    return std.math.cast(usize, x.integer) orelse error.BadConfig;
}

fn number(v: std.json.Value, key: []const u8) !f32 {
    const x = v.object.get(key) orelse return error.BadConfig;
    return switch (x) {
        .integer => @floatFromInt(x.integer),
        .float => @floatCast(x.float),
        else => error.BadConfig,
    };
}

pub fn tokens(frames: usize) usize {
    const tail = frames % 100;
    return frames / 100 * 13 + (tail + 7) / 8;
}

test "audio output length keeps each hundred-frame convolution chunk's own padding" {
    try std.testing.expectEqual(@as(usize, 0), tokens(0));
    try std.testing.expectEqual(@as(usize, 13), tokens(100));
    try std.testing.expectEqual(@as(usize, 14), tokens(101));
    try std.testing.expectEqual(@as(usize, 390), tokens(3000));
}
