const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;

pub const Linear = struct { w: u64, b: u64 = 0 };
pub const Norm = struct { w: u64, b: u64 = 0 };
pub const EncoderLayer = struct { norm: Norm, q: Linear, k: Linear, v: Linear, o: Linear, post: Norm, up: Linear, down: Linear };
pub const DecoderLayer = struct { norm: u64, q: u64, k: u64, v: u64, qnorm: u64, knorm: u64, o: u64, post: u64, gate: u64, up: u64, down: u64 };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    c: Config,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    conv: [3]Linear = undefined,
    conv_out: u64 = 0,
    enc: [24]EncoderLayer = undefined,
    enc_norm: Norm = undefined,
    proj1: Linear = undefined,
    proj2: Linear = undefined,
    embed: u64 = 0,
    dec: [28]DecoderLayer = undefined,
    norm: u64 = 0,
    head: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
    }
};

const Loader = struct {
    w: *Weights,
    ck: *core.Checkpoint,
    d: *const cuda.Driver,

    fn one(l: *Loader, name: []const u8, shape: []const usize) !u64 {
        const t = try l.ck.expect(name, .bf16, shape);
        var b = try cuda.DeviceBuffer.alloc(l.d, t.bytes.len);
        errdefer b.free();
        try b.upload(0, t.bytes);
        try l.w.buffers.append(l.w.gpa, b);
        return b.ptr;
    }

    fn linear(l: *Loader, prefix: []const u8, n: usize, k: usize) !Linear {
        var b: [256]u8 = undefined;
        const w = try l.one(try std.fmt.bufPrint(&b, "{s}.weight", .{prefix}), &.{ n, k });
        return .{ .w = w, .b = try l.one(try std.fmt.bufPrint(&b, "{s}.bias", .{prefix}), &.{n}) };
    }

    fn norm(l: *Loader, prefix: []const u8, n: usize) !Norm {
        var b: [256]u8 = undefined;
        const w = try l.one(try std.fmt.bufPrint(&b, "{s}.weight", .{prefix}), &.{n});
        return .{ .w = w, .b = try l.one(try std.fmt.bufPrint(&b, "{s}.bias", .{prefix}), &.{n}) };
    }
};

pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const path = try std.fs.path.join(arena.allocator(), &.{ dir, "config.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena.allocator(), .limited(1 << 20));
    var w: Weights = .{ .gpa = a, .c = try Config.parse(arena.allocator(), bytes) };
    errdefer w.deinit();
    var ck = try core.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    var l: Loader = .{ .w = &w, .ck = &ck, .d = d };
    var buf: [256]u8 = undefined;
    for (&w.conv, 0..) |*v, i| {
        v.w = try l.one(try std.fmt.bufPrint(&buf, "thinker.audio_tower.conv2d{d}.weight", .{i + 1}), &.{ 480, if (i == 0) 1 else 480, 3, 3 });
        v.b = try l.one(try std.fmt.bufPrint(&buf, "thinker.audio_tower.conv2d{d}.bias", .{i + 1}), &.{480});
    }
    w.conv_out = try l.one("thinker.audio_tower.conv_out.weight", &.{ 1024, 7680 });
    for (&w.enc, 0..) |*e, i| {
        const prefix = try std.fmt.allocPrint(arena.allocator(), "thinker.audio_tower.layers.{d}", .{i});
        e.norm = try l.norm(try std.fmt.bufPrint(&buf, "{s}.self_attn_layer_norm", .{prefix}), 1024);
        e.post = try l.norm(try std.fmt.bufPrint(&buf, "{s}.final_layer_norm", .{prefix}), 1024);
        e.q = try l.linear(try std.fmt.bufPrint(&buf, "{s}.self_attn.q_proj", .{prefix}), 1024, 1024);
        e.k = try l.linear(try std.fmt.bufPrint(&buf, "{s}.self_attn.k_proj", .{prefix}), 1024, 1024);
        e.v = try l.linear(try std.fmt.bufPrint(&buf, "{s}.self_attn.v_proj", .{prefix}), 1024, 1024);
        e.o = try l.linear(try std.fmt.bufPrint(&buf, "{s}.self_attn.out_proj", .{prefix}), 1024, 1024);
        e.up = try l.linear(try std.fmt.bufPrint(&buf, "{s}.fc1", .{prefix}), 4096, 1024);
        e.down = try l.linear(try std.fmt.bufPrint(&buf, "{s}.fc2", .{prefix}), 1024, 4096);
    }
    w.enc_norm = try l.norm("thinker.audio_tower.ln_post", 1024);
    w.proj1 = try l.linear("thinker.audio_tower.proj1", 1024, 1024);
    w.proj2 = try l.linear("thinker.audio_tower.proj2", 2048, 1024);
    w.embed = try l.one("thinker.model.embed_tokens.weight", &.{ w.c.vocab, 2048 });
    w.head = try l.one("thinker.lm_head.weight", &.{ w.c.vocab, 2048 });
    w.norm = try l.one("thinker.model.norm.weight", &.{2048});
    for (&w.dec, 0..) |*e, i| {
        const prefix = try std.fmt.allocPrint(arena.allocator(), "thinker.model.layers.{d}", .{i});
        e.norm = try l.one(try std.fmt.bufPrint(&buf, "{s}.input_layernorm.weight", .{prefix}), &.{2048});
        e.post = try l.one(try std.fmt.bufPrint(&buf, "{s}.post_attention_layernorm.weight", .{prefix}), &.{2048});
        e.q = try l.one(try std.fmt.bufPrint(&buf, "{s}.self_attn.q_proj.weight", .{prefix}), &.{ 2048, 2048 });
        e.k = try l.one(try std.fmt.bufPrint(&buf, "{s}.self_attn.k_proj.weight", .{prefix}), &.{ 1024, 2048 });
        e.v = try l.one(try std.fmt.bufPrint(&buf, "{s}.self_attn.v_proj.weight", .{prefix}), &.{ 1024, 2048 });
        e.o = try l.one(try std.fmt.bufPrint(&buf, "{s}.self_attn.o_proj.weight", .{prefix}), &.{ 2048, 2048 });
        e.qnorm = try l.one(try std.fmt.bufPrint(&buf, "{s}.self_attn.q_norm.weight", .{prefix}), &.{128});
        e.knorm = try l.one(try std.fmt.bufPrint(&buf, "{s}.self_attn.k_norm.weight", .{prefix}), &.{128});
        e.gate = try l.one(try std.fmt.bufPrint(&buf, "{s}.mlp.gate_proj.weight", .{prefix}), &.{ 6144, 2048 });
        e.up = try l.one(try std.fmt.bufPrint(&buf, "{s}.mlp.up_proj.weight", .{prefix}), &.{ 6144, 2048 });
        e.down = try l.one(try std.fmt.bufPrint(&buf, "{s}.mlp.down_proj.weight", .{prefix}), &.{ 2048, 6144 });
    }
    if (ck.unused() != 0) return error.UnusedWeights;
    return w;
}
