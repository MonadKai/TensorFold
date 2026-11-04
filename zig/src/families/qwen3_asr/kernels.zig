const std = @import("std");
const cuda = @import("cuda");

pub const Kernels = struct {
    mod: cuda.Module,
    im2col_fn: cuda.Function,
    pack_fn: cuda.Function,
    attention_fn: cuda.Function,
    vit: cuda.vit.Kernels,
    gemms: cuda.vit.Gemms,
    norms: cuda.norms.Norms,
    embed: cuda.embed_rows.EmbedRows,
    qk: cuda.qk_rows.QkRows,
    attn: cuda.window_attention.WindowAttention,
    act: cuda.act_rows.ActRows,
    sample_mod: cuda.Module,
    round_mod: cuda.Module,
    draws: cuda.sample.Draws,
    round: cuda.tree_round.Kernels,

    pub fn load(a: std.mem.Allocator, d: *const cuda.Driver) !Kernels {
        var k: Kernels = undefined;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        k.mod = try cuda.Module.load(d, cuda.kernels.qwen3_asr);
        errdefer k.mod.unload();
        k.im2col_fn = try k.mod.function("tf_asr_im2col");
        k.pack_fn = try k.mod.function("tf_asr_conv_pack");
        k.attention_fn = try k.mod.function("tf_asr_audio_attention");
        k.vit = try cuda.vit.Kernels.load(d);
        errdefer k.vit.unload();
        k.gemms = try cuda.vit.Gemms.init(a, d);
        errdefer k.gemms.deinit();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer k.qk.unload();
        k.attn = try cuda.window_attention.WindowAttention.load(d);
        errdefer k.attn.unload();
        k.act = try cuda.act_rows.ActRows.load(d);
        errdefer k.act.unload();
        k.sample_mod = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer k.sample_mod.unload();
        k.round_mod = try cuda.Module.load(d, cuda.kernels.tree_round);
        errdefer k.round_mod.unload();
        k.draws = try cuda.sample.Draws.resolve(k.sample_mod);
        k.round = try cuda.tree_round.Kernels.resolve(k.round_mod);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.round_mod.unload();
        k.sample_mod.unload();
        k.act.unload();
        k.attn.unload();
        k.qk.unload();
        k.embed.unload();
        k.norms.unload();
        k.gemms.deinit();
        k.vit.unload();
        k.mod.unload();
    }

    pub fn im2col(k: *const Kernels, s: cuda.Stream, input: u64, out: u64, height: usize, width: usize, channels: usize, frames: usize, start: usize, mel: bool) !void {
        var args: cuda.launch.Args = .{};
        args.add(input);
        args.add(out);
        for ([_]usize{ height, width, channels, frames, start, @intFromBool(mel) }) |v| args.add(@as(i32, @intCast(v)));
        const count = ((height + 1) / 2) * ((width + 1) / 2) * channels * 9;
        try cuda.launch.launch(k.im2col_fn, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, s, &args);
    }

    pub fn pack(k: *const Kernels, s: cuda.Stream, input: u64, out: u64, time: usize) !void {
        var args: cuda.launch.Args = .{};
        args.add(input);
        args.add(out);
        for ([_]usize{ time, 16, 480 }) |v| args.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(k.pack_fn, .{ .grid = .{ .x = @intCast((time * 7680 + 255) / 256) }, .block = .{ .x = 256 } }, s, &args);
    }

    pub fn audioAttention(k: *const Kernels, s: cuda.Stream, q: u64, key: u64, v: u64, out: u64, rows: usize, window: usize) !void {
        if (window == 0 or window > 128) return error.BadAudioWindow;
        var args: cuda.launch.Args = .{};
        for ([_]u64{ q, key, v, out }) |p| args.add(p);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(window)));
        try cuda.launch.launch(k.attention_fn, .{ .grid = .{ .x = @intCast(rows), .y = 16 }, .block = .{ .x = 128 } }, s, &args);
    }
};
