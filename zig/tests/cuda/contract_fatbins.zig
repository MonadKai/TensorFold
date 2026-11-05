//! Offline CUDA compile receipts; no device or driver context is created.
const std = @import("std");
const contract = @import("core").contract;
const limits = contract.limits;
const registry = contract.registry;
const check = @import("check.zig");
fn records(what: []const u8, rel: ?limits.Nvcc, set: limits.Nvcc) void {
    if (rel) |v| {
        const own = std.meta.eql(v, set);
        std.debug.print("INFO limits {s}: images built by nvcc {d}.{d}, checked against nvcc {d}.{d}'s records{s}\n", .{ what, v.major, v.minor, set.major, set.minor, if (own) "" else " (the newest set at or below it)" });
    } else std.debug.print("INFO limits {s}: the images name no nvcc release; checked against nvcc {d}.{d}'s records\n", .{ what, set.major, set.minor });
}

/// kernel-contract-limits DIR: each record against DIR's fatbins; a RECORD line keeps the record for SMs not built.
fn keep(a: std.mem.Allocator, line: *std.ArrayList(u8), r: registry.CudaRecord, compiler: ?limits.Nvcc, i: usize) !void {
    const u = registry.usageFor(r, compiler)[i];
    try line.print(a, ".{{ .regs = {d}, .static = {d}, .bound = {d} }}, ", .{ u.regs, u.static, u.bound });
}

const Cached = struct {
    bytes: []u8,
    images: []contract.cubin.Image,
    kernels: [5]?[]contract.cubin.Kernel = @splat(null),

    fn load(a: std.mem.Allocator, io: std.Io, path: []const u8) !Cached {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
        errdefer a.free(bytes);
        return .{ .bytes = bytes, .images = try contract.cubin.images(a, bytes) };
    }

    fn deinit(cache: *Cached, a: std.mem.Allocator) void {
        for (cache.kernels) |kernels| if (kernels) |items| a.free(items);
        contract.cubin.freeImages(a, cache.images);
        a.free(cache.bytes);
    }
};

pub fn run(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    var cache: std.StringHashMapUnmanaged(Cached) = .empty;
    defer {
        var it = cache.valueIterator();
        while (it.next()) |value| value.deinit(a);
        cache.deinit(a);
    }
    var n: usize = 0;
    var said: ?limits.Nvcc = null;
    for (registry.all_cuda ++ registry.cuda_controls) |r| {
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        const any_target = for (limits.sms, 0..) |target, index| {
            if (registry.targetDecision(r, target, r.usage[index]) == .applicable) break true;
        } else false;
        if (!control and !any_target) {
            std.debug.print("N/A limits {s}: no applicable target\n", .{r.launch.name});
            continue;
        }
        const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ dir, r.launch.image });
        defer a.free(path);
        const entry = try cache.getOrPut(a, r.launch.image);
        if (!entry.found_existing) entry.value_ptr.* = Cached.load(a, io, path) catch |e| {
            _ = cache.remove(r.launch.image);
            return e;
        };
        const cached = entry.value_ptr;
        const imgs = cached.images;
        const compiler: ?limits.Nvcc = if (imgs.len > 0) try contract.cubin.release(a, imgs[0].elf) else null;
        var red: usize = 0;
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(a);
        for (limits.sms, 0..) |t, i| {
            const decision = registry.targetDecision(r, t, r.usage[i]);
            if (!control and decision != .applicable) {
                std.debug.print("N/A limits {s} sm_{d}: {t}\n", .{ r.launch.name, t.sm, decision });
                try keep(a, &line, r, compiler, i);
                continue;
            }
            const img = for (imgs) |im| {
                if (im.sm == t.sm) break im;
            } else {
                std.debug.print("FAIL limits {s}: {s} holds no sm_{d} image (build with -Dsm=80,86,89,120,121)\n", .{ r.launch.name, path, t.sm });
                n += 1;
                try keep(a, &line, r, compiler, i);
                continue;
            };
            if (cached.kernels[i] == null) cached.kernels[i] = try contract.cubin.kernels(a, img.elf, t.sm);
            const ks = cached.kernels[i].?;
            const k = for (ks) |kv| {
                if (std.mem.eql(u8, kv.name, r.launch.symbol)) break kv;
            } else return error.SymbolNotInImage;
            const rel = try contract.cubin.release(a, img.elf);
            const set = registry.setFor(rel);
            if (said == null or !std.meta.eql(said.?, set)) records("fatbins", rel, set);
            said = set;
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(a);
            const measured = registry.targetDecision(r, t, k.usage);
            if (!control and measured != .applicable) {
                std.debug.print("N/A limits {s} sm_{d}: {t}\n", .{ r.launch.name, t.sm, measured });
                try keep(a, &line, r, compiler, i);
                continue;
            }
            red += try limits.cuda(r.launch, k.usage, t, &out, a);
            const current = std.meta.eql(k.usage, registry.usageFor(r, rel)[i]);
            if (!control and !current) n += 1;
            std.debug.print("{s} limits sm_{d} {s}: {d} registers, {d} static shared bytes, bound {d} (record {s}){s}", .{ if (out.items.len > 0) (if (control) "RED " else "FAIL") else "PASS", t.sm, r.launch.name, k.usage.regs, k.usage.static, k.usage.bound, if (current) "current" else "stale", if (out.items.len > 0) ": " else "\n" });
            if (out.items.len > 0) std.debug.print("{s}", .{out.items});
            try line.print(a, ".{{ .regs = {d}, .static = {d}, .bound = {d} }}, ", .{ k.usage.regs, k.usage.static, k.usage.bound });
        }
        std.debug.print("RECORD {s}: .usage = .{{ {s}}}\n", .{ r.launch.name, line.items });
        if (control and red == 0) n += 1;
        if (!control) n += red;
    }
    try check.expect(n == 0, "kernel-contract-limits: {d} problems", .{n});
    check.pass("kernel-contract-limits: every registered launch fits sm_80, sm_86, sm_89, sm_120 and sm_121, every control red", .{});
}
