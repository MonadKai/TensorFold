const std = @import("std");

pub const Audio = struct {
    pub const DecodeBatches = struct { stable: [8]u64, throughput: [8]u64, row_capacity: usize };
    pub const Cpu = struct { decode_seconds: f64 = 0, resample_seconds: f64 = 0, log_mel_seconds: f64 = 0 };
    cpu: ?Cpu = null,
    features: []const f32,
    frames: u32,
    at: u32,
    tokens: u32,
    hash: [32]u8,

    pub fn valid(audio: Audio, prompt: []const u32, token: u32) bool {
        if (audio.frames == 0 or audio.frames > 60000 or audio.features.len != @as(usize, audio.frames) * 128 or
            audio.tokens == 0 or audio.at > prompt.len or audio.tokens > prompt.len - audio.at) return false;
        for (audio.features) |v| if (!std.math.isFinite(v)) return false;
        for (prompt[audio.at..][0..audio.tokens]) |v| if (v != token) return false;
        return true;
    }
};
