"""
debug_subprocess.py
Test whether worker subprocesses can start and produce data.
Isolates the SubprocessGenerator from all DFL training logic.
"""
import multiprocessing
import sys


def worker_func(param):
    """Generator that runs in a child subprocess."""
    import sys
    print(f"[worker pid={__import__('os').getpid()}] starting", flush=True)

    # Test imports that DFL workers need
    print("[worker] importing numpy...", flush=True)
    import numpy as np
    print("[worker] importing cv2...", flush=True)
    import cv2
    print("[worker] importing scipy...", flush=True)
    import scipy.linalg
    print("[worker] all imports OK", flush=True)

    for i in range(5):
        print(f"[worker] yielding batch {i}", flush=True)
        yield np.zeros((2, 64, 64, 3), dtype=np.float32)
    print("[worker] done", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    print(f"[main pid={__import__('os').getpid()}] starting", flush=True)
    print(f"[main] Python {sys.version}", flush=True)

    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

    from core.joblib.SubprocessGenerator import SubprocessGenerator

    print("[main] creating SubprocessGenerator...", flush=True)
    gen = SubprocessGenerator(worker_func, user_param=None, prefetch=2, start_now=True)

    print("[main] waiting for first batch (30s timeout)...", flush=True)
    try:
        batch = next(gen)
        print(f"[main] got batch shape={batch.shape}", flush=True)
        batch2 = next(gen)
        print(f"[main] got batch2 shape={batch2.shape}", flush=True)
        print("[main] SUCCESS — subprocess workers are functional", flush=True)
    except RuntimeError as e:
        print(f"[main] WORKER ERROR: {e}", flush=True)
    except Exception as e:
        import traceback
        print(f"[main] EXCEPTION: {traceback.format_exc()}", flush=True)
