const std = @import("std");
const Complex = std.math.Complex(f32);

pub const bins = 128;
pub const hop = 160;
pub const fft_size = 400;

fn mel(frequency: f64) f64 {
    return if (frequency < 1000) frequency / (200.0 / 3.0) else 15 + @log(frequency / 1000) / (@log(6.4) / 27.0);
}

fn hz(m: f64) f64 {
    return if (m < 15) m * (200.0 / 3.0) else 1000 * @exp((m - 15) * (@log(6.4) / 27.0));
}

pub const Plan = struct {
    window: [fft_size]f32,
    twiddle: [fft_size]Complex,
    filters: [bins][201]f32,

    pub fn init() Plan {
        var p: Plan = undefined;
        for (0..fft_size) |i| {
            const angle = 2 * std.math.pi * @as(f64, @floatFromInt(i)) / fft_size;
            p.window[i] = @floatCast(0.5 - 0.5 * @cos(angle));
            p.twiddle[i] = .{ .re = @floatCast(@cos(angle)), .im = @floatCast(-@sin(angle)) };
        }
        for (&p.filters, 0..) |*row, b| {
            const low = hz(mel(8000) * @as(f64, @floatFromInt(b)) / 129);
            const mid = hz(mel(8000) * @as(f64, @floatFromInt(b + 1)) / 129);
            const high = hz(mel(8000) * @as(f64, @floatFromInt(b + 2)) / 129);
            for (row, 0..) |*v, k| {
                const f = @as(f64, @floatFromInt(k)) * 40;
                v.* = @floatCast(@max(0, @min((f - low) / (mid - low), (high - f) / (high - mid))) * 2 / (high - low));
            }
        }
        return p;
    }

    fn fft(p: *const Plan, input: []const f32, stride: usize, out: []Complex) void {
        const n = out.len;
        if (n == 1) {
            out[0] = .{ .re = input[0], .im = 0 };
            return;
        }
        const radix: usize = if (n % 2 == 0) 2 else 5;
        const m = n / radix;
        var tmp: [fft_size]Complex = undefined;
        for (0..radix) |r| p.fft(input[r * stride ..], stride * radix, tmp[r * m ..][0..m]);
        for (out, 0..) |*v, k| {
            var re: f32 = 0;
            var im: f32 = 0;
            for (0..radix) |r| {
                const z = tmp[r * m + k % m];
                const w = p.twiddle[(r * k * (fft_size / n)) % fft_size];
                re += z.re * w.re - z.im * w.im;
                im += z.re * w.im + z.im * w.re;
            }
            v.* = .{ .re = re, .im = im };
        }
    }

    pub fn extract(p: *const Plan, a: std.mem.Allocator, samples: []const f32) ![]f32 {
        if (samples.len <= fft_size / 2) return error.AudioTooShort;
        for (samples) |v| if (!std.math.isFinite(v)) return error.NonFiniteAudio;
        const frames = samples.len / hop;
        const out = try a.alloc(f32, bins * frames);
        errdefer a.free(out);
        var peak: f32 = -std.math.inf(f32);
        for (0..frames) |t| {
            var frame: [fft_size]f32 = undefined;
            for (&frame, 0..) |*v, j| {
                var at: isize = @as(isize, @intCast(t * hop + j)) - fft_size / 2;
                if (at < 0) at = -at;
                if (at >= samples.len) at = 2 * @as(isize, @intCast(samples.len)) - 2 - at;
                v.* = samples[@intCast(at)] * p.window[j];
            }
            var freq: [fft_size]Complex = undefined;
            p.fft(&frame, 1, &freq);
            var power: [201]f32 = undefined;
            for (&power, freq[0..201]) |*v, z| v.* = z.re * z.re + z.im * z.im;
            for (&p.filters, 0..) |*filter, b| {
                var sum: f32 = 0;
                for (filter, power) |w, x| sum += w * x;
                const value = @log10(@max(sum, 1e-10));
                out[b * frames + t] = value;
                peak = @max(peak, value);
            }
        }
        for (out) |*v| v.* = (@max(v.*, peak - 8) + 4) / 4;
        return out;
    }
};

test "silence is finite at the Whisper floor and short/nonfinite clips refuse" {
    const p = Plan.init();
    const x: [1600]f32 = @splat(0);
    const features = try p.extract(std.testing.allocator, &x);
    defer std.testing.allocator.free(features);
    try std.testing.expectEqual(@as(usize, 1280), features.len);
    for (features) |v| try std.testing.expectEqual(@as(f32, -1.5), v);
    try std.testing.expectError(error.AudioTooShort, p.extract(std.testing.allocator, &.{0}));
    var bad = x;
    bad[3] = std.math.nan(f32);
    try std.testing.expectError(error.NonFiniteAudio, p.extract(std.testing.allocator, &bad));
}

test "mixed-radix Fourier output agrees with an independent float64 DFT" {
    const p = Plan.init();
    var input: [fft_size]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13) + 0.2 * @cos(@as(f32, @floatFromInt(i)) * 0.7);
    var got: [fft_size]Complex = undefined;
    p.fft(&input, 1, &got);
    for (got, 0..) |v, k| {
        var real: f64 = 0;
        var imag: f64 = 0;
        for (input, 0..) |x, i| {
            const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k * i)) / fft_size;
            real += x * @cos(angle);
            imag += x * @sin(angle);
        }
        try std.testing.expectApproxEqAbs(real, @as(f64, v.re), 0.0001);
        try std.testing.expectApproxEqAbs(imag, @as(f64, v.im), 0.0001);
    }
}
