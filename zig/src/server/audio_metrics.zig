//! Optional audio pipeline timings; shared batch intervals are not additive across requests.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");

pub fn put(a: std.mem.Allocator, runtime: anytype, stats: api.Stats, cpu_opt: ?api.Audio.Cpu, postprocess_seconds: f64) !void {
    try runtime.put(a, "queued_seconds", .{ .float = stats.queued_seconds });
    if (cpu_opt != null) try runtime.put(a, "postprocess_seconds", .{ .float = postprocess_seconds });
    if (cpu_opt) |cpu| {
        const times = try json.newObject(a);
        inline for (@typeInfo(@TypeOf(cpu)).@"struct".field_names) |name| try times.put(a, name, .{ .float = @field(cpu, name) });
        try runtime.put(a, "audio_cpu_intervals", .{ .object = times });
    }
    if (stats.audio_time) |timing| {
        const stages = try json.newObject(a);
        inline for (@typeInfo(@TypeOf(timing)).@"struct".field_names) |name| {
            const value = @field(timing, name);
            try stages.put(a, name, if (@TypeOf(value) == f64) .{ .float = value } else try json.intValue(a, value));
        }
        try runtime.put(a, "audio_gpu_event_intervals", .{ .object = stages });
    }
}
