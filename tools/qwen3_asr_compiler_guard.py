"""Observe Linux CUDA/Zig compiler processes and require a quiet interval before timing."""
import subprocess
import time


def snapshot():
    result = subprocess.run(["ps", "-C", "zig,nvcc,cicc", "-o", "pid=,comm="], capture_output=True, text=True)
    if result.returncode not in (0, 1) or result.stderr:
        raise RuntimeError("compiler process observation failed")
    return [dict(pid=int(line.split()[0]), name=line.split()[1]) for line in result.stdout.splitlines() if line.strip()]


def wait_quiet(seconds, timeout, probe=snapshot, now=time.monotonic, sleep=time.sleep):
    start = now()
    quiet_since = start
    observations = []
    while True:
        active = probe()
        elapsed = now() - start
        observations.append(dict(elapsed_seconds=elapsed, processes=active))
        if active:
            quiet_since = None
        else:
            if quiet_since is None:
                quiet_since = now()
            if now() - quiet_since >= seconds:
                return observations
        if elapsed >= timeout:
            raise TimeoutError(observations)
        sleep(min(1, seconds))
