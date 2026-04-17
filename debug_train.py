"""
debug_train.py
Minimal SAEHD training proof-of-concept with no interactive prompts.
All parameters are hard-coded to the smallest/fastest possible values.

Usage:
    .\\venv\\Scripts\\python.exe debug_train.py

NOTE: patch functions must be at module level (not inside if __name__ == "__main__")
so that multiprocessing spawn can pickle them when spawning worker subprocesses.
"""

import multiprocessing


# ── patch stubs ───────────────────────────────────────────────────────────────
# Defined at module level so spawn can pickle them.

def _no_input_in_time(prompt, timeout):
    """Replaces io.input_in_time — always returns False (no override)."""
    return False

def _no_input_skip_pending():
    """Replaces io.input_skip_pending — no-op, avoids spawning a stdin-drain subprocess."""
    pass


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    import argparse
    import pickle
    from pathlib import Path

    # ── args ───────────────────────────────────────────────────────────────────
    # Defaults derive from this file's location so the script still works when
    # invoked directly without arguments.
    _ROOT = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Headless SAEHD training")
    ap.add_argument("--model-name",      default="poc",
                    help="Model name prefix (default: poc)")
    ap.add_argument("--model-dir",       default=str(_ROOT / "workspace" / "model"),
                    help="Directory where model files are stored")
    ap.add_argument("--src-aligned-dir", default=str(_ROOT / "workspace" / "data_src" / "aligned"),
                    help="Source aligned faces directory")
    ap.add_argument("--dst-aligned-dir", default=str(_ROOT / "workspace" / "data_dst" / "aligned"),
                    help="Destination aligned faces directory")
    ap.add_argument("--pretrained-dir",  default=None,
                    help="Path to pretrained pack directory. "
                         "Default: <model-dir>/../pretrained/SAEHD/")
    args = ap.parse_args()

    from core.leras import nn
    nn.initialize_main_env()

    # ── paths ──────────────────────────────────────────────────────────────────
    MODEL_DIR = Path(args.model_dir)
    SRC_DIR   = Path(args.src_aligned_dir)
    DST_DIR   = Path(args.dst_aligned_dir)

    MODEL_NAME  = args.model_name   # internally becomes "<model_name>_SAEHD"
    MODEL_CLASS = "SAEHD"

    # ── smallest/fastest SAEHD options (cold-start defaults) ──────────────────
    # These are only used when no pretrained pack is found.
    # If a compatible pretrained pack exists, its architecture settings replace
    # the resolution/archi/dims values below; see bootstrap section.
    OPTIONS = {
        "resolution"           : 64,      # minimum allowed
        "face_type"            : "f",     # full face
        "archi"                : "df",    # simplest architecture
        "ae_dims"              : 32,      # minimum
        "e_dims"               : 16,      # minimum
        "d_dims"               : 16,      # minimum
        "d_mask_dims"          : 16,      # minimum (must be even)
        "batch_size"           : 2,       # smallest useful batch
        "masked_training"      : False,
        "eyes_mouth_prio"      : False,
        "uniform_yaw"          : False,
        "blur_out_mask"        : False,
        "adabelief"            : True,
        "lr_dropout"           : "n",
        "random_warp"          : True,
        "random_hsv_power"     : 0.0,
        "gan_power"            : 0.0,     # no GAN — removes discriminator overhead
        "gan_patch_size"       : 8,
        "gan_dims"             : 16,
        "true_face_power"      : 0.0,
        "face_style_power"     : 0.0,
        "bg_style_power"       : 0.0,
        "ct_mode"              : "none",
        "clipgrad"             : False,
        "pretrain"             : False,
        "models_opt_on_gpu"    : True,
        "autobackup_hour"      : 0,
        "write_preview_history": False,
        "target_iter"          : 0,
        "random_src_flip"      : False,
        "random_dst_flip"      : True,
    }

    # ── seed the model data file ───────────────────────────────────────────────
    # ModelBase builds: force_model_name + "_" + model_class_name = "poc_SAEHD"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    internal_name = f"{MODEL_NAME}_{MODEL_CLASS}"
    data_path     = MODEL_DIR / f"{internal_name}_data.dat"

    # ── pretrained bootstrap ───────────────────────────────────────────────────
    # Check ordering:
    #   1. Weights already present → existing model, skip all seeding.
    #   2. Pretrained pack available → copy weights, adopt pack's architecture.
    #   3. Otherwise → cold start with OPTIONS defaults above.
    from pretrained_bootstrap import model_has_weights, try_bootstrap

    if model_has_weights(MODEL_DIR, internal_name):
        # ── Case 1: resume existing model ────────────────────────────────────
        # Weights are on disk; data.dat already reflects the correct options.
        # Do NOT overwrite either — that would corrupt the session architecture.
        print(f"[bootstrap] Model weights found in {MODEL_DIR} — resuming existing model.",
              flush=True)
        ACTIVE_OPTIONS = None  # sentinel: skip seeding below

    else:
        # ── Case 2/3: new session — try pretrained bootstrap ─────────────────
        if args.pretrained_dir is not None:
            pretrained_dir = Path(args.pretrained_dir)
        else:
            # Convention: <model_dir>/../pretrained/SAEHD/
            # e.g. workspace/model/../pretrained/SAEHD = workspace/pretrained/SAEHD
            pretrained_dir = MODEL_DIR.parent / "pretrained" / MODEL_CLASS

        bootstrap_opts = try_bootstrap(MODEL_DIR, internal_name, pretrained_dir)
        if bootstrap_opts is not None:
            # Merge: pretrained architecture settings override the cold-start
            # defaults, but non-architecture training options (batch_size,
            # random_warp, etc.) stay as defined in OPTIONS above.
            ACTIVE_OPTIONS = {**OPTIONS, **{
                k: bootstrap_opts[k]
                for k in ("resolution", "archi", "ae_dims", "e_dims",
                           "d_dims", "d_mask_dims", "face_type")
                if k in bootstrap_opts
            }}
        else:
            # ── Case 3: cold start ────────────────────────────────────────────
            ACTIVE_OPTIONS = OPTIONS

    # ── write seed files (skipped on resume) ──────────────────────────────────
    if ACTIVE_OPTIONS is not None:
        model_data = {
            "iter"               : 1,       # non-zero → skips all first-run prompts
            "options"            : ACTIVE_OPTIONS,
            "loss_history"       : [[0.5, 0.5]],
            "sample_for_preview" : None,
            "choosed_gpu_indexes": None,
        }
        data_path.write_bytes(pickle.dumps(model_data))
        print(f"[setup] Wrote model seed -> {data_path}", flush=True)

        default_opts_path = MODEL_DIR / f"{MODEL_CLASS}_default_options.dat"
        default_opts_path.write_bytes(pickle.dumps(ACTIVE_OPTIONS))
        print(f"[setup] Wrote default options -> {default_opts_path}", flush=True)

    # ── patch io ───────────────────────────────────────────────────────────────
    from core.interact import interact as io
    io.input_in_time    = _no_input_in_time     # ask_override() never waits
    io.input_skip_pending = _no_input_skip_pending  # no stdin-drain subprocess

    # ── run ────────────────────────────────────────────────────────────────────
    from mainscripts import Trainer

    Trainer.main(
        model_class_name       = MODEL_CLASS,
        saved_models_path      = MODEL_DIR,
        training_data_src_path = SRC_DIR,
        training_data_dst_path = DST_DIR,
        pretraining_data_path  = None,
        pretrained_model_path  = None,
        no_preview             = True,
        force_model_name       = MODEL_NAME,
        force_gpu_idxs         = [0],
        cpu_only               = False,
        silent_start           = False,
        execute_programs       = [],
        debug                  = False,
    )
