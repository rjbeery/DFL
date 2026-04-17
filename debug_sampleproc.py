"""
debug_sampleproc.py
Tests SampleProcessor.process in a subprocess with real face data.
This exactly replicates what batch_func does.
"""
import multiprocessing
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def worker_func(samples, index_cli, output_sample_types, sample_process_options):
    pid = os.getpid()
    print(f"[worker {pid}] starting", flush=True)

    from samplelib import SampleProcessor
    from facelib import FaceType
    import numpy as np

    print(f"[worker {pid}] getting 2 indices...", flush=True)
    indexes = index_cli.multi_get(2)
    print(f"[worker {pid}] got indices: {indexes}", flush=True)

    for n in range(2):
        sample = samples[indexes[n]]
        print(f"[worker {pid}] processing sample {n}: {sample.filename}", flush=True)
        try:
            x, = SampleProcessor.process(
                [sample], sample_process_options, output_sample_types, False, ct_sample=None
            )
            print(f"[worker {pid}] sample {n} processed OK, shapes={[a.shape for a in x]}", flush=True)
        except Exception as e:
            import traceback
            print(f"[worker {pid}] FAILED on sample {n}: {traceback.format_exc()}", flush=True)

    print(f"[worker {pid}] done", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    from pathlib import Path
    from core.leras import nn
    nn.initialize_main_env()

    from samplelib import SampleLoader, SampleProcessor, SampleType
    from facelib import FaceType
    from core.mplib import IndexHost

    ROOT    = Path(__file__).parent
    SRC_DIR = ROOT / "workspace" / "data_src" / "aligned"

    print("[main] loading samples...", flush=True)
    samples = SampleLoader.load(SampleType.FACE, SRC_DIR)
    print(f"[main] loaded {len(samples)} samples", flush=True)

    host = IndexHost(len(samples))
    cli  = host.create_cli()

    output_sample_types = [
        {'sample_type': SampleProcessor.SampleType.FACE_IMAGE, 'warp': True,  'transform': True,
         'channel_type': SampleProcessor.ChannelType.BGR, 'ct_mode': None,
         'random_hsv_shift_amount': 0.0, 'face_type': FaceType.FULL,
         'data_format': 'NHWC', 'resolution': 64},
        {'sample_type': SampleProcessor.SampleType.FACE_IMAGE, 'warp': False, 'transform': True,
         'channel_type': SampleProcessor.ChannelType.BGR, 'ct_mode': None,
         'face_type': FaceType.FULL,
         'data_format': 'NHWC', 'resolution': 64},
    ]
    sample_process_options = SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=False)

    print("[main] spawning worker subprocess...", flush=True)
    p = multiprocessing.Process(
        target=worker_func,
        args=(samples, cli, output_sample_types, sample_process_options)
    )
    p.daemon = True
    p.start()

    print("[main] waiting (30s)...", flush=True)
    p.join(timeout=30)

    if p.is_alive():
        print("[main] TIMEOUT — worker hung. SampleProcessor is the problem.", flush=True)
        p.terminate(); p.join()
    else:
        if p.exitcode == 0:
            print("[main] SUCCESS — SampleProcessor works in subprocess.", flush=True)
        else:
            print(f"[main] Worker crashed with exit code {p.exitcode}", flush=True)
