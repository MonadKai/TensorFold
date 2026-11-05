//! Target admission and compiler version ordering for CUDA records.
const limits = @import("limits.zig");
pub const TargetDecision = enum { applicable, below_minimum, optional_shared_limit };
pub fn targetDecision(r: anytype, t: limits.Sm, u: limits.Usage) TargetDecision {
    if (t.sm < r.launch.min_sm) return .below_minimum;
    if (r.optional_shared and !@import("gpu_memory").fits(u.static, r.launch.dynamic, t.smem_optin)) return .optional_shared_limit;
    return .applicable;
}

pub fn newer(x: limits.Nvcc, y: limits.Nvcc) bool {
    return x.major > y.major or (x.major == y.major and x.minor > y.minor);
}
