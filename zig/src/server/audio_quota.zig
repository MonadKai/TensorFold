//! Fair parent admission retains half the lanes for ordinary requests; shrinking a quota never releases work.
const std = @import("std");

pub const Parent = struct {
    node: std.DoublyLinkedList.Node = .{},
    active: u32 = 0,
    waiting: u32 = 0,
    last_grant: u64 = 0,
    registered: bool = false,
};
const Waiter = struct { node: std.DoublyLinkedList.Node = .{}, parent: *Parent };

pub const Gate = struct {
    mutex: std.Io.Mutex = .init,
    queue: std.DoublyLinkedList = .{},
    parent_list: std.DoublyLinkedList = .{},
    active: u32 = 0,
    waiting: u32 = 0,
    peak: u32 = 0,
    parents: u32 = 0,
    turn: u64 = 0,

    pub fn limit(lanes: u32) u32 {
        return @max(lanes / 2, 1);
    }

    pub fn register(g: *Gate, io: std.Io, parent: *Parent) void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        std.debug.assert(!parent.registered and parent.active == 0 and parent.waiting == 0);
        parent.node = .{};
        parent.last_grant = g.turn;
        parent.registered = true;
        g.parent_list.append(&parent.node);
        g.parents += 1;
    }

    pub fn unregister(g: *Gate, io: std.Io, parent: *Parent) void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        std.debug.assert(parent.registered and parent.active == 0 and parent.waiting == 0);
        g.parent_list.remove(&parent.node);
        g.parents -= 1;
        parent.registered = false;
    }

    fn enqueue(g: *Gate, waiter: *Waiter) void {
        std.debug.assert(waiter.parent.registered);
        g.queue.append(&waiter.node);
        g.waiting += 1;
        waiter.parent.waiting += 1;
    }

    fn remove(g: *Gate, waiter: *Waiter) void {
        g.queue.remove(&waiter.node);
        g.waiting -= 1;
        waiter.parent.waiting -= 1;
    }

    fn eligible(g: *Gate, capacity: u32) ?*Waiter {
        if (g.parents == 0 or g.active >= capacity) return null;
        const share = capacity / g.parents + @intFromBool(capacity % g.parents != 0);
        var chosen: ?*Waiter = null;
        var oldest: u64 = 0;
        var node = g.queue.first;
        while (node) |item| : (node = item.next) {
            const waiter: *Waiter = @fieldParentPtr("node", item);
            const age = g.turn -% waiter.parent.last_grant;
            if (waiter.parent.active < share and (chosen == null or age > oldest)) {
                chosen = waiter;
                oldest = age;
            }
        }
        return chosen;
    }

    fn take(g: *Gate, waiter: *Waiter, capacity: u32) bool {
        if (g.eligible(capacity) != waiter) return false;
        g.remove(waiter);
        g.active += 1;
        waiter.parent.active += 1;
        g.turn +%= 1;
        waiter.parent.last_grant = g.turn;
        g.peak = @max(g.peak, g.active);
        return true;
    }

    pub fn wait(g: *Gate, io: std.Io, capacity: u32, parent: *Parent, gone: anytype) error{Cancelled}!void {
        var waiter: Waiter = .{ .parent = parent };
        g.mutex.lockUncancelable(io);
        g.enqueue(&waiter);
        g.mutex.unlock(io);
        while (true) {
            g.mutex.lockUncancelable(io);
            if (gone.check()) {
                g.remove(&waiter);
                g.mutex.unlock(io);
                return error.Cancelled;
            }
            const admitted = g.take(&waiter, capacity);
            g.mutex.unlock(io);
            if (admitted) return;
            std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
        }
    }

    pub fn release(g: *Gate, io: std.Io, parent: *Parent) void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        std.debug.assert(g.active > 0 and parent.active > 0);
        g.active -= 1;
        parent.active -= 1;
    }

    pub fn render(g: *Gate, io: std.Io, w: *std.Io.Writer, lanes: u32) !void {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        inline for (.{ "active", "waiting", "peak", "parents" }) |name|
            try w.print("# TYPE tensorfold:audio_chunks_{s} gauge\ntensorfold:audio_chunks_{s} {d}\n", .{ name, name, @field(g, name) });
        try w.print("# TYPE tensorfold:audio_chunks_limit gauge\ntensorfold:audio_chunks_limit {d}\n", .{limit(lanes)});
        var maximum: u32 = 0;
        var node = g.parent_list.first;
        while (node) |item| : (node = item.next) {
            const parent: *Parent = @fieldParentPtr("node", item);
            maximum = @max(maximum, parent.active);
        }
        try w.print("# TYPE tensorfold:audio_parent_active_max gauge\ntensorfold:audio_parent_active_max {d}\n", .{maximum});
    }
};

test "later parents receive a fair turn before earlier parents refill" {
    var gate: Gate = .{};
    var parents: [2]Parent = @splat(.{});
    for (&parents) |*parent| gate.register(std.testing.io, parent);
    var waiters = [_]Waiter{ .{ .parent = &parents[0] }, .{ .parent = &parents[0] }, .{ .parent = &parents[1] }, .{ .parent = &parents[1] } };
    for (&waiters) |*waiter| gate.enqueue(waiter);
    try std.testing.expect(gate.take(&waiters[0], 4));
    try std.testing.expect(!gate.take(&waiters[1], 4));
    try std.testing.expect(gate.take(&waiters[2], 4));
    try std.testing.expect(gate.take(&waiters[1], 4));
    try std.testing.expect(gate.take(&waiters[3], 4));
    for (&parents) |*parent| {
        gate.release(std.testing.io, parent);
        gate.release(std.testing.io, parent);
        gate.unregister(std.testing.io, parent);
    }
    try std.testing.expectEqual(@as(u32, 0), gate.active);
    try std.testing.expectEqual(@as(u32, 0), gate.parents);
}

test "shrinking a fair share keeps occupied children until completion" {
    var gate: Gate = .{};
    var first: Parent = .{};
    gate.register(std.testing.io, &first);
    var waiters: [4]Waiter = @splat(.{ .parent = &first });
    for (&waiters) |*waiter| {
        gate.enqueue(waiter);
        try std.testing.expect(gate.take(waiter, 4));
    }
    var later: Parent = .{};
    gate.register(std.testing.io, &later);
    var pending: Waiter = .{ .parent = &later };
    gate.enqueue(&pending);
    try std.testing.expectEqual(@as(u32, 4), first.active);
    try std.testing.expect(!gate.take(&pending, 2));
    gate.release(std.testing.io, &first);
    try std.testing.expect(!gate.take(&pending, 2));
    gate.release(std.testing.io, &first);
    try std.testing.expect(!gate.take(&pending, 2));
    gate.release(std.testing.io, &first);
    try std.testing.expect(gate.take(&pending, 2));
    gate.release(std.testing.io, &first);
    gate.release(std.testing.io, &later);
    gate.unregister(std.testing.io, &first);
    gate.unregister(std.testing.io, &later);
}

test "cancelled waiters leave no parent or global admission state" {
    var gate: Gate = .{};
    var parent: Parent = .{};
    gate.register(std.testing.io, &parent);
    const Gone = struct {
        fn check(_: @This()) bool {
            return true;
        }
    };
    try std.testing.expectError(error.Cancelled, gate.wait(std.testing.io, 1, &parent, Gone{}));
    gate.unregister(std.testing.io, &parent);
    try std.testing.expectEqual(@as(u32, 0), gate.waiting);
    try std.testing.expect(gate.queue.first == null);
}
