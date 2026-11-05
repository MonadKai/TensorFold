const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const layout = @import("contract_asr_layout.zig");
const attention = @import("contract_asr_attention.zig");
const blas = @import("contract_blas_rows.zig");

pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    layouts: [4]layout.Layout,
    attentions: [3]attention.Attention,
    gemms: cuda.vit.Gemms,
    projection: blas.Projection,
    dense_rows: ?cuda.row_gemm.Rows = null,
    dense_projection: ?blas.Projection = null,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(core.contract.gates.Kernel)) !*Group {
        const group = try a.create(Group);
        errdefer a.destroy(group);
        group.a = a;
        group.mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen3_asr);
        errdefer group.mod.unload();
        group.gemms = try cuda.vit.Gemms.init(a, gpu.d);
        errdefer group.gemms.deinit();
        group.projection = .{ .rig = .{ .gpu = gpu, .s = s }, .gemms = &group.gemms };
        group.dense_rows = null;
        group.dense_projection = null;
        if (try gpu.d.currentCapability() == 121 and group.gemms.lt.api.cublasLtGetVersion() == 130100) {
            group.dense_rows = try cuda.row_gemm.Rows.init(gpu.d, &group.gemms.lt);
            group.dense_projection = .{ .rig = .{ .gpu = gpu, .s = s }, .gemms = &group.gemms, .dense = &group.dense_rows.? };
        }
        errdefer if (group.dense_rows) |*rows| rows.deinit();
        const im2col = try group.mod.function("tf_asr_im2col");
        const pack = try group.mod.function("tf_asr_conv_pack");
        const attn = try group.mod.function("tf_asr_audio_attention");
        for (&group.layouts, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .f = if (i < 2) im2col else pack,
            .kind = if (i == 0) .mel else if (i == 1) .conv else .pack,
            .fault = i == 3,
        };
        for (&group.attentions, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .f = attn,
            .fault = i == 2,
            .qk_stress = i == 1,
        };
        try kernels.ensureUnusedCapacity(a, 9);
        kernels.appendAssumeCapacity(group.projection.kernel());
        if (group.dense_projection) |*projection| kernels.appendAssumeCapacity(projection.kernel());
        for (&group.layouts) |*item| kernels.appendAssumeCapacity(item.kernel());
        for (&group.attentions) |*item| kernels.appendAssumeCapacity(item.kernel());
        return group;
    }

    pub fn destroy(group: *Group) void {
        for (&group.layouts) |*item| item.deinit();
        for (&group.attentions) |*item| item.deinit();
        group.projection.deinit();
        if (group.dense_projection) |*projection| projection.deinit();
        if (group.dense_rows) |*rows| rows.deinit();
        group.gemms.deinit();
        group.mod.unload();
        group.a.destroy(group);
    }
};
