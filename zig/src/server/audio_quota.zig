//! FIFO child admission reserves half the lanes for ordinary requests when at least two lanes exist.
const std = @import("std");

pub const Gate = struct {
    mutex: std.Io.Mutex = .init,
    queue: std.DoublyLinkedList = .{},
    active: u32 = 0,
    waiting: u32 = 0,
    peak: u32 = 0,

    pub fn limit(lanes: u32) u32 {
        return @max(lanes / 2, 1);
    }

    fn enqueue(g: *Gate, node: *std.DoublyLinkedList.Node) void {
        g.queue.append(node);
        g.waiting += 1;
    }

    fn remove(g: *Gate, node: *std.DoublyLinkedList.Node) void {
        g.queue.remove(node);
        g.waiting -= 1;
    }

    fn take(g: *Gate, node: *std.DoublyLinkedList.Node, capacity: u32) bool {
        if (g.queue.first != node or g.active >= capacity) return false;
        g.remove(node);
        g.active += 1;
        g.peak = @max(g.peak, g.active);
        return true;
    }

    pub fn wait(g: *Gate, io: std.Io, capacity: u32, gone: anytype) error{Cancelled}!void {
        var node: std.DoublyLinkedList.Node = .{};
        g.mutex.lockUncancelable(io);
        g.enqueue(&node);
        g.mutex.unlock(io);
        while (true) {
            g.mutex.lockUncancelable(io);
            if (gone.check()) {
                g.remove(&node);
                g.mutex.unlock(io);
                return error.Cancelled;
            }
            const admitted = g.take(&node, capacity);
            g.mutex.unlock(io);
            if (admitted) return;
            std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
        }
    }

    pub fn release(g: *Gate, io: std.Io) void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        std.debug.assert(g.active > 0);
        g.active -= 1;
    }

    pub fn render(g: *Gate, io: std.Io, w: *std.Io.Writer, lanes: u32) !void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        inline for (.{ "active", "waiting", "peak" }) |name|
            try w.print("# TYPE tensorfold:audio_chunks_{s} gauge\ntensorfold:audio_chunks_{s} {d}\n", .{ name, name, @field(g, name) });
        try w.print("# TYPE tensorfold:audio_chunks_limit gauge\ntensorfold:audio_chunks_limit {d}\n", .{limit(lanes)});
    }
};

test "audio child quota keeps FIFO order and cancellation releases its queued place" {
    var gate: Gate = .{};
    var nodes: [3]std.DoublyLinkedList.Node = @splat(.{});
    for (&nodes) |*node| gate.enqueue(node);
    try std.testing.expect(!gate.take(&nodes[1], 1));
    try std.testing.expect(gate.take(&nodes[0], 1));
    try std.testing.expect(!gate.take(&nodes[1], 1));
    gate.remove(&nodes[1]);
    gate.release(std.testing.io);
    try std.testing.expect(gate.take(&nodes[2], 1));
    gate.release(std.testing.io);
    try std.testing.expectEqual(@as(u32, 0), gate.active);
    try std.testing.expectEqual(@as(u32, 0), gate.waiting);
    try std.testing.expectEqual(@as(u32, 1), gate.peak);
    try std.testing.expectEqual(@as(u32, 4), Gate.limit(8));
}

test "audio child quota removes a cancelled waiter before admitting it" {
    var gate: Gate = .{};
    const Gone = struct {
        fn check(_: @This()) bool {
            return true;
        }
    };
    try std.testing.expectError(error.Cancelled, gate.wait(std.testing.io, 1, Gone{}));
    try std.testing.expectEqual(@as(u32, 0), gate.waiting);
    try std.testing.expect(gate.queue.first == null);
}
