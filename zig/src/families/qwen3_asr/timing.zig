const std = @import("std");
const cuda = @import("cuda");
pub const Sum = @import("lanes").Stream.AudioTime;
pub const Stage = enum { encoder, prefill, decode, head, sample };
const Pair = struct { begin: cuda.Event, end: cuda.Event, used: bool = false };

pub const Timing = struct {
    pairs: [5]Pair,
    rows: usize = 0,

    pub fn init(d: *const cuda.Driver) !?Timing {
        if (std.c.getenv("TF_ASR_PROFILE") == null) return null;
        var result: Timing = undefined;
        result.rows = 0;
        var count: usize = 0;
        errdefer for (result.pairs[0..count]) |*pair| {
            pair.begin.deinit();
            pair.end.deinit();
        };
        for (&result.pairs) |*pair| {
            var begin = try cuda.Event.init(d, true);
            errdefer begin.deinit();
            pair.* = .{ .begin = begin, .end = try cuda.Event.init(d, true) };
            count += 1;
        }
        return result;
    }

    pub fn deinit(t: *Timing) void {
        for (&t.pairs) |*pair| {
            pair.begin.deinit();
            pair.end.deinit();
        }
    }

    pub fn start(t: *Timing, s: cuda.Stream, stage: Stage) !void {
        const p = &t.pairs[@backingInt(stage)];
        if (p.used) return error.UnresolvedTiming;
        try p.begin.record(s);
    }

    pub fn end(t: *Timing, s: cuda.Stream, stage: Stage) !void {
        const p = &t.pairs[@backingInt(stage)];
        try p.end.record(s);
        p.used = true;
    }

    pub fn resolve(t: *Timing) !Sum {
        var result: Sum = .{};
        inline for (std.enums.values(Stage)) |stage| {
            const p = &t.pairs[@backingInt(stage)];
            if (p.used) {
                const seconds: f64 = @as(f64, try cuda.Event.elapsedMs(p.begin, p.end)) / 1000;
                @field(result, @tagName(stage) ++ "_seconds") = seconds;
                p.used = false;
                if (stage == .decode) {
                    result.decode_calls = 1;
                    result.decode_rows = t.rows;
                }
            }
        }
        return result;
    }
};
