"""Observe Linux CUDA/Zig compiler processes and require a quiet interval before timing."""
import subprocess
import time
from pathlib import Path


class ObservationError(RuntimeError):
    def __init__(self, message, observations):
        super().__init__(message)
        self.observations = list(observations)


def snapshot(container_id=None):
    result = subprocess.run(["ps", "-C", "zig,nvcc,cicc,ptxas,cudafe++,fatbinary,nvlink", "-o", "pid=,comm="], capture_output=True, text=True)
    if result.returncode not in (0, 1) or result.stderr:
        raise RuntimeError("compiler process observation failed")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, name = line.split()
        try:
            in_scope = container_id is None or container_id in (Path("/proc") / pid / "cgroup").read_text()
        except FileNotFoundError:
            continue
        rows.append(dict(pid=int(pid), name=name, in_scope=in_scope))
    return rows


def wait_quiet(seconds, timeout, probe=snapshot, now=time.monotonic, sleep=time.sleep):
    start = now()
    quiet_since = start
    observations = []
    while True:
        try:
            active = probe()
        except Exception as error:
            raise ObservationError(str(error), observations) from error
        elapsed = now() - start
        observations.append(dict(elapsed_seconds=elapsed, processes=active))
        if any(row.get("in_scope", True) for row in active):
            quiet_since = None
        else:
            if quiet_since is None:
                quiet_since = now()
            if now() - quiet_since >= seconds:
                return observations
        if elapsed >= timeout:
            raise TimeoutError(observations)
        sleep(min(1, seconds))
