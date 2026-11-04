const std = @import("std");
const cuda = @import("cuda");
const Model = @import("model.zig").Model;
const Linear = @import("weights.zig").Linear;
const tokens = @import("config.zig").tokens;

pub const max_frames = 60000;
pub const max_rows = 7800;

pub const Encoder = struct {
    features: u64,
    col: u64,
    conv: [2]u64,
    pack: u64,
    position: u64,
    x: u64,
    norm: u64,
    q: u64,
    k: u64,
    v: u64,
    att: u64,
    tmp: u64,
    up: u64,
    acc: u64,
    out: u64,

    pub fn init(m: *Model) !Encoder {
        const r = max_rows;
        const e: Encoder = .{ .features = try m.alloc(128 * max_frames * 4), .col = try m.alloc(32 * 25 * 480 * 9 * 2), .conv = .{ try m.alloc(64 * 50 * 480 * 2), try m.alloc(64 * 50 * 480 * 2) }, .pack = try m.alloc(13 * 7680 * 2), .position = try m.alloc(13 * 1024 * 2), .x = try m.alloc(r * 1024 * 2), .norm = try m.alloc(r * 1024 * 2), .q = try m.alloc(r * 1024 * 2), .k = try m.alloc(r * 1024 * 2), .v = try m.alloc(r * 1024 * 2), .att = try m.alloc(r * 1024 * 2), .tmp = try m.alloc(r * 1024 * 2), .up = try m.alloc(r * 4096 * 2), .acc = try m.alloc(r * 4096 * 4), .out = try m.alloc(r * 2048 * 2) };
        var pos: [13 * 1024]u16 = undefined;
        for (0..13) |t| for (0..512) |j| {
            const freq = @exp(-@log(@as(f32, 10000)) * @as(f32, @floatFromInt(j)) / 511);
            const angle = @as(f32, @floatFromInt(t)) * freq;
            pos[t * 1024 + j] = cuda.vit.bf16(@sin(angle));
            pos[t * 1024 + 512 + j] = cuda.vit.bf16(@cos(angle));
        };
        try m.upload(e.position, std.mem.sliceAsBytes(&pos));
        return e;
    }

    fn linear(e: *Encoder, m: *Model, l: Linear, input: u64, out: u64, r: usize, n: usize, k: usize) !void {
        try cuda.vit.linear(&m.k.gemms, &m.k.vit, m.s, e.acc, input, l.w, l.b, out, r, n, k);
    }

    pub fn run(e: *Encoder, m: *Model, features: []const f32, frames: usize) !u64 {
        if (frames == 0 or frames > max_frames or features.len != 128 * frames) return error.BadAudioFeatures;
        try m.upload(e.features, std.mem.sliceAsBytes(features));
        try m.tap("audio.mel", e.features, 128, frames, false);
        const chunk_width: usize = @min(frames, 100);
        const chunk_time: usize = (chunk_width + 7) / 8;
        var out_at: usize = 0;
        var start: usize = 0;
        while (start < frames) : (start += 100) {
            var height: usize = 128;
            var width = chunk_width;
            var channels: usize = 1;
            var input = e.features;
            for (m.w.conv, 0..) |l, i| {
                try m.k.im2col(m.s, input, e.col, height, width, channels, frames, start, i == 0);
                const oh = (height + 1) / 2;
                const ow = (width + 1) / 2;
                const out = e.conv[i % 2];
                try e.linear(m, l, e.col, out, oh * ow, 480, channels * 9);
                try m.k.vit.pointwise(m.s, out, null, out, oh * ow * 480, .gelu);
                input = out;
                height = oh;
                width = ow;
                channels = 480;
            }
            try m.k.pack(m.s, input, e.pack, chunk_time);
            try m.k.gemms.run(m.s, e.pack, m.w.conv_out, e.tmp, chunk_time, 1024, 7680, .bf16);
            try m.k.vit.pointwise(m.s, e.tmp, e.position, e.tmp, chunk_time * 1024, .none);
            const kept = tokens(@min(100, frames - start));
            try m.copy(e.x + out_at * 2048, e.tmp, kept * 2048);
            out_at += kept;
        }
        const r = out_at;
        try m.tap("audio.conv", e.x, r, 1024, true);
        for (m.w.enc, 0..) |l, i| {
            try m.k.vit.layerNorm(m.s, e.x, l.norm.w, l.norm.b, e.norm, r, 1024, 1e-5);
            try e.linear(m, l.q, e.norm, e.q, r, 1024, 1024);
            try e.linear(m, l.k, e.norm, e.k, r, 1024, 1024);
            try e.linear(m, l.v, e.norm, e.v, r, 1024, 1024);
            try m.k.audioAttention(m.s, e.q, e.k, e.v, e.att, r, chunk_time * 8);
            try e.linear(m, l.o, e.att, e.tmp, r, 1024, 1024);
            try m.k.vit.pointwise(m.s, e.tmp, e.x, e.x, r * 1024, .none);
            try m.k.vit.layerNorm(m.s, e.x, l.post.w, l.post.b, e.norm, r, 1024, 1e-5);
            try e.linear(m, l.up, e.norm, e.up, r, 4096, 1024);
            try m.k.vit.pointwise(m.s, e.up, null, e.up, r * 4096, .gelu);
            try e.linear(m, l.down, e.up, e.tmp, r, 1024, 4096);
            try m.k.vit.pointwise(m.s, e.tmp, e.x, e.x, r * 1024, .none);
            if (i == 0 or i == 23) {
                var label: [64]u8 = undefined;
                try m.tap(try std.fmt.bufPrint(&label, "audio.layer{d}", .{i}), e.x, r, 1024, true);
            }
        }
        try m.k.vit.layerNorm(m.s, e.x, m.w.enc_norm.w, m.w.enc_norm.b, e.norm, r, 1024, 1e-5);
        try e.linear(m, m.w.proj1, e.norm, e.tmp, r, 1024, 1024);
        try m.k.vit.pointwise(m.s, e.tmp, null, e.tmp, r * 1024, .gelu);
        try e.linear(m, m.w.proj2, e.tmp, e.out, r, 2048, 1024);
        try m.tap("audio.embeddings", e.out, r, 2048, true);
        return e.out;
    }
};
