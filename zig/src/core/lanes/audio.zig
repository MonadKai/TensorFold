const std = @import("std");

pub const Audio = struct {
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
