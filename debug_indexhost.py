"""
debug_indexhost.py
Tests whether IndexHost.Cli can communicate across spawn subprocess boundary.
This is the exact mechanism batch_func uses to get sample indices.
"""
import multiprocessing
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def worker_func(index_cli):
    """Runs in subprocess — tries to get indices via IndexHost.Cli."""
    pid = os.getpid()
    print(f"[worker {pid}] starting multi_get...", flush=True)
    try:
        result = index_cli.multi_get(2)
        print(f"[worker {pid}] got indices: {result}", flush=True)
    except Exception as e:
        import traceback
        print(f"[worker {pid}] ERROR: {traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    from core.mplib import IndexHost

    print(f"[main {os.getpid()}] creating IndexHost(100)...", flush=True)
    host = IndexHost(100)

    print("[main] calling create_cli()...", flush=True)
    cli = host.create_cli()

    print("[main] spawning subprocess...", flush=True)
    p = multiprocessing.Process(target=worker_func, args=(cli,))
    p.daemon = True
    p.start()

    print("[main] waiting for subprocess (10s)...", flush=True)
    p.join(timeout=10)

    if p.is_alive():
        print("[main] TIMEOUT — subprocess still alive, killing it", flush=True)
        p.terminate()
        p.join()
        print("[main] CONCLUSION: IndexHost.Cli cross-process communication BROKEN", flush=True)
    else:
        print(f"[main] subprocess exited with code {p.exitcode}", flush=True)
        if p.exitcode == 0:
            print("[main] CONCLUSION: IndexHost.Cli works fine across processes", flush=True)
        else:
            print("[main] CONCLUSION: subprocess crashed", flush=True)
