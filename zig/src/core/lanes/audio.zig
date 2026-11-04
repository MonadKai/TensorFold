pub const Audio = struct {
    features: []const f32,
    frames: u32,
    at: u32,
    tokens: u32,
    hash: [32]u8,
};
