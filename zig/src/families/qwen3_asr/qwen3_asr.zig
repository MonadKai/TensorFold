const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const weights = @import("weights.zig");
const Kernels = @import("kernels.zig").Kernels;
const Model = @import("model.zig").Model;
const Backend = @import("backend.zig").Backend;

pub const model_type = "qwen3_asr";
pub const formats: []const []const u8 = &.{"bf16"};
pub const default_context: i64 = 8192;
pub const max_segments: u32 = 1;
pub const prompt_rows = @import("model.zig").rows;
pub const explains_open = true;
pub const audio_input = true;
pub const decodeBatches = @import("model.zig").decodeBatches;
pub const takes_drafter = false;
pub const drafter_refusal = "Qwen3-ASR has no draft head in this CUDA slice";

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null };
pub const Loaded = struct { backend: lanes.backend.Backend, facts: lanes.Model, rows: u32, stream_bytes: usize, free_streams: u32, ctx: *anyopaque, deinit: *const fn (*anyopaque) void, target: *Backend };
const Owned = struct { a: std.mem.Allocator, w: weights.Weights, k: Kernels, s: cuda.Stream, m: Model, b: Backend };

pub fn contextCeiling(a: std.mem.Allocator, io: std.Io, dir: []const u8, yarn: ?f64) !i64 {
    if (yarn != null) return error.UnsupportedRopeScaling;
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20));
    return @intCast((try @import("config.zig").Config.parse(a, bytes)).context);
}

pub fn open(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: Options) !Loaded {
    if (captured != null) return error.CapturedKernels;
    if (o.context > 65536 or o.context == 0) return error.BadContext;
    if (o.lanes_fixed and (o.lanes == 0 or o.lanes > 8)) return error.ParallelLimit;
    try @import("core").audio_decode.available();
    const own = try a.create(Owned);
    errdefer a.destroy(own);
    own.a = a;
    own.w = try weights.load(a, io, ctx.d, dir);
    errdefer own.w.deinit();
    own.k = try Kernels.load(a, ctx.d);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    const room = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const per = Model.streamBytes(o.context) + @import("encoder.zig").max_rows * 4096;
    const fit = (room -| (1 << 30)) / per;
    if (fit == 0) return error.NoStreamFits;
    if (o.lanes_fixed and o.lanes > fit) return error.ParallelMemory;
    const count: usize = @intCast(@min(@min(@max(o.lanes, 1), 8), fit));
    own.m = try Model.init(a, ctx.d, own.s, &own.w, &own.k, o.context, count);
    errdefer own.m.deinit();
    own.b = try Backend.init(&own.m);
    errdefer own.b.deinit();
    try own.s.synchronize();
    return .{ .backend = own.b.backend(), .facts = own.b.facts(), .rows = 32, .stream_bytes = per, .free_streams = @intCast(count), .ctx = own, .deinit = close, .target = &own.b };
}

fn close(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.s.synchronize() catch {};
    own.b.deinit();
    own.m.deinit();
    own.s.deinit();
    own.k.deinit();
    own.w.deinit();
    own.a.destroy(own);
}

pub fn explain(_: ?*anyopaque, e: anyerror) ?[]const u8 {
    return switch (e) {
        error.OneAudioRequired => "Qwen3-ASR requires exactly one audio per request",
        error.BadAudioFeatures => "Qwen3-ASR accepts finite audio between 201 samples and 600 seconds",
        error.ParallelLimit => "Qwen3-ASR CUDA admits 1 to 8 streams",
        error.NoStreamFits, error.ParallelMemory => "Qwen3-ASR weights, audio scratch and stream caches exceed the CUDA budget",
        error.CapturedKernels => "Qwen3-ASR uses native CUDA kernels; unset TENSORFOLD_CUDA_KERNELS",
        error.InvalidAsrArithmeticMode => "TENSORFOLD_QWEN3_ASR_MATH must be stable or throughput",
        error.UnqualifiedRowGemmRuntime, error.UnqualifiedRowGemmAlgorithm => "ASR throughput requires its qualified SM 121 and cuBLASLt 130100 policy; select stable explicitly on another runtime",
        error.BadContext => "Qwen3-ASR context must be between 1 and 65536 tokens",
        error.UnsupportedCheckpointSchema => "this Qwen3-ASR slice loads the original checkpoint with thinker_config",
        error.AudioLibraryUnavailable => "Qwen3-ASR requires libtfaudio.so; run zig build audio with FFmpeg development libraries",
        error.UnsupportedQwen3AsrGeometry => "this Qwen3-ASR CUDA slice qualifies the original BF16 1.7B geometry only",
        else => null,
    };
}

test {
    _ = @import("config.zig");
    _ = @import("core").audio_features;
}
