"""
debug_merge.py
--------------
Headless DFL merge — runs the merger non-interactively, accepting all default
merger settings from the trained model's saved config.

All interactive prompts in Merger.py and MergerConfig.ask_settings() are
handled by patching the interact (io) module so every question returns its
default value.  This avoids needing a display or terminal input.

Strategy:
  - io.input_bool  → always False   (selects non-interactive merge mode)
  - io.input_int   → default value  (accept all numeric defaults)
  - io.input_str   → default value  (accept all string defaults)
  - io.input_in_time / input_skip_pending → no-ops

This means the merger runs with whatever settings are stored in the model's
merger_session.dat (if it exists from a previous interactive run), or uses the
model's built-in defaults on first run.

NOTE: patch functions must be at module level (not inside if __name__ == "__main__")
so that multiprocessing spawn can pickle them when spawning worker subprocesses.

Invoked by command_registry.py as a subprocess:
    python -u debug_merge.py --model-class SAEHD --model-name poc \\
        --model-dir /workspace/model \\
        --input-dir /workspace/data_dst \\
        --output-dir /workspace/output \\
        --output-mask-dir /workspace/output_mask \\
        --aligned-dir /workspace/data_dst/aligned
"""

import multiprocessing


# ── patch stubs (module-level for spawn pickling) ─────────────────────────────

def _false_input_bool(s, default_value, help_message=None):
    """
    Always returns False.
    When called for 'Use interactive merger?', False selects non-interactive
    mode so the merger runs to completion without a preview window.
    For any other bool prompt, False is a safe conservative choice.
    """
    print(f"[merge-auto] {s}: n (non-interactive)", flush=True)
    return False


def _default_input_int(s, default_value, valid_range=None, valid_list=None,
                       add_info=None, show_default_value=True, help_message=None):
    """Returns the default value for every integer prompt."""
    print(f"[merge-auto] {s}: {default_value} (default)", flush=True)
    return default_value


def _default_input_str(s, default_value=None, valid_list=None,
                       show_default_value=True, help_message=None):
    """Returns the default value for every string prompt."""
    print(f"[merge-auto] {s}: {default_value!r} (default)", flush=True)
    return default_value


def _default_input_number(s, default_value, valid_list=None,
                           show_default_value=True, add_info=None, help_message=None):
    """Returns the default value for every numeric prompt."""
    print(f"[merge-auto] {s}: {default_value} (default)", flush=True)
    return default_value


def _no_input_in_time(prompt, timeout):
    """No-op: always reports 'no input received'."""
    return False


def _no_input_skip_pending():
    """No-op: avoids spawning a stdin-drain subprocess."""
    pass


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    import argparse
    from pathlib import Path

    _ROOT = Path(__file__).parent

    ap = argparse.ArgumentParser(description="Headless DFL merge")
    ap.add_argument(
        "--model-class",
        default="SAEHD",
        help="DFL model class name, e.g. SAEHD, AMP, Quick96  (default: SAEHD)",
    )
    ap.add_argument(
        "--model-name",
        default="poc",
        help="Force model name — must match the prefix used during training  (default: poc)",
    )
    ap.add_argument(
        "--model-dir",
        default=str(_ROOT / "workspace" / "model"),
        help="Directory containing trained model files  (default: workspace/model)",
    )
    ap.add_argument(
        "--input-dir",
        default=str(_ROOT / "workspace" / "data_dst"),
        help="Raw destination frames to merge into  (default: workspace/data_dst)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(_ROOT / "workspace" / "output"),
        help="Destination for merged output frames  (default: workspace/output)",
    )
    ap.add_argument(
        "--output-mask-dir",
        default=str(_ROOT / "workspace" / "output_mask"),
        help="Destination for merge mask images  (default: workspace/output_mask)",
    )
    ap.add_argument(
        "--aligned-dir",
        default=str(_ROOT / "workspace" / "data_dst" / "aligned"),
        help="Aligned destination faces directory  (default: workspace/data_dst/aligned)",
    )
    args = ap.parse_args()

    # Print resolved paths so they appear in the SSE log stream.
    print(f"[merge] model class   : {args.model_class}",    flush=True)
    print(f"[merge] model name    : {args.model_name}",     flush=True)
    print(f"[merge] model dir     : {args.model_dir}",      flush=True)
    print(f"[merge] input dir     : {args.input_dir}",      flush=True)
    print(f"[merge] output dir    : {args.output_dir}",     flush=True)
    print(f"[merge] mask dir      : {args.output_mask_dir}", flush=True)
    print(f"[merge] aligned dir   : {args.aligned_dir}",    flush=True)

    from core.leras import nn
    nn.initialize_main_env()

    # ── patch interact so all prompts accept their default values ─────────────
    from core.interact import interact as io
    io.input_bool         = _false_input_bool
    io.input_int          = _default_input_int
    io.input_str          = _default_input_str
    io.input_number       = _default_input_number
    io.input_in_time      = _no_input_in_time
    io.input_skip_pending = _no_input_skip_pending

    # ── run ───────────────────────────────────────────────────────────────────
    from mainscripts import Merger

    Merger.main(
        model_class_name  = args.model_class,
        saved_models_path = Path(args.model_dir),
        force_model_name  = args.model_name,
        input_path        = Path(args.input_dir),
        output_path       = Path(args.output_dir),
        output_mask_path  = Path(args.output_mask_dir),
        aligned_path      = Path(args.aligned_dir),
        force_gpu_idxs    = [0],
        cpu_only          = False,
    )
