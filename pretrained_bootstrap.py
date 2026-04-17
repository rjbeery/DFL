"""
pretrained_bootstrap.py
-----------------------
Optional pretrained-weight bootstrap for debug_train.py.

When model_dir contains no weight files, checks for a pretrained pack at
pretrained_dir.  If found and valid, copies the weight files into model_dir
so the training session warm-starts from pretrained weights instead of
random init.

Called by debug_train.py before the data.dat seed step.
Not called when model weights already exist (resume path).

Public API
----------
model_has_weights(model_dir, model_name) -> bool
    Check whether this session already has weight files on disk.

try_bootstrap(model_dir, model_name, pretrained_dir) -> dict | None
    Attempt bootstrap.  Returns the pretrained options dict on success
    (caller uses it for data.dat seeding), or None if bootstrap was skipped.

Pretrained pack layout (pretrained_dir/)
-----------------------------------------
  data.dat          pickle of {"options": {...}, "iter": N, ...}
                    Only "options" is read; iter/loss_history are ignored.
  encoder.npy       \
  inter.npy          | df architecture weights
  decoder_src.npy    |
  decoder_dst.npy   /
  -- or --
  encoder.npy       \
  inter_AB.npy       | liae architecture weights
  inter_B.npy        |
  decoder.npy       /

Architecture settings adopted from the pack
-------------------------------------------
When bootstrap succeeds the session uses the pack's architecture settings
(resolution, archi, ae_dims, e_dims, d_dims, d_mask_dims) rather than the
defaults hard-coded in debug_train.py.  Other training-behaviour options
(batch_size, random_warp, adabelief, etc.) continue to use the values from
debug_train.py.OPTIONS.

Compatibility rules
-------------------
Bootstrap is skipped (with a logged reason) if:
  - pretrained_dir does not exist
  - data.dat is missing or unreadable
  - options fail basic sanity checks (resolution out of range, unknown archi)
  - any expected weight file is absent

In all skip cases training continues with a normal cold-start.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

# Architecture options that determine network layer shapes.
# Inherited from the pretrained pack; must be internally consistent.
_ARCH_KEYS = ("resolution", "archi", "ae_dims", "e_dims", "d_dims", "d_mask_dims")

# Weight files required for each SAEHD architecture variant.
_DF_WEIGHTS   = ("encoder.npy", "inter.npy", "decoder_src.npy", "decoder_dst.npy")
_LIAE_WEIGHTS = ("encoder.npy", "inter_AB.npy", "inter_B.npy", "decoder.npy")

_VALID_ARCHI_TYPES = {"df", "liae"}

_LOG = "[bootstrap]"


# ── Public API ────────────────────────────────────────────────────────────────

def model_has_weights(model_dir: Path, model_name: str) -> bool:
    """
    Return True if the encoder weight file already exists in model_dir.

    The encoder is always the first weight file regardless of archi, so its
    presence is a reliable indicator that this session has been initialised
    (either from a previous run or a prior bootstrap).
    """
    return (model_dir / f"{model_name}_encoder.npy").exists()


def _find_pack_prefix(pretrained_dir: Path) -> Optional[str]:
    """
    Detect the filename prefix used in a pretrained pack directory.

    Packs exported directly from a DFL model directory carry the model name
    as a prefix on every file (e.g. '256wf_SAEHD_encoder.npy').  Bare packs
    have no prefix ('encoder.npy').

    Returns '' for bare files, '<prefix>' for prefixed files, or None if the
    encoder weight cannot be found at all.
    """
    if (pretrained_dir / "encoder.npy").exists():
        return ""
    matches = list(pretrained_dir.glob("*encoder.npy"))
    if len(matches) == 1:
        # Strip the trailing 'encoder.npy' to get the prefix, e.g. '256wf_SAEHD_'
        return matches[0].name[: -len("encoder.npy")]
    return None


def try_bootstrap(
    model_dir:      Path,
    model_name:     str,
    pretrained_dir: Path,
) -> Optional[dict]:
    """
    Attempt to bootstrap model_dir from the pretrained pack in pretrained_dir.

    Steps:
      1. Confirm pretrained_dir exists and locate its file prefix.
      2. Read and sanity-check the pack's options from data.dat.
      3. Determine which weight files are needed for the pack's archi.
      4. Confirm all weight files are present.
      5. Copy weight files into model_dir with the session name prefix.

    Returns the pretrained options dict on success so the caller can seed
    data.dat with the correct architecture settings.
    Returns None on any failure; caller falls back to cold-start defaults.
    """
    # ── 1. Pretrained directory & prefix detection ────────────────────────────
    if not pretrained_dir.exists():
        print(f"{_LOG} No pretrained pack at {pretrained_dir} — cold start.", flush=True)
        return None

    prefix = _find_pack_prefix(pretrained_dir)
    if prefix is None:
        print(f"{_LOG} Pretrained pack at {pretrained_dir} has no encoder.npy — skipping.",
              flush=True)
        return None

    pack_data_path = pretrained_dir / f"{prefix}data.dat"
    if not pack_data_path.exists():
        print(f"{_LOG} Pretrained pack at {pretrained_dir} is missing data.dat — skipping.",
              flush=True)
        return None

    # ── 2. Read and validate options ──────────────────────────────────────────
    try:
        pack_data = pickle.loads(pack_data_path.read_bytes())
        pack_opts: dict = pack_data.get("options", {})
    except Exception as exc:
        print(f"{_LOG} Could not read pretrained data.dat: {exc} — skipping.", flush=True)
        return None

    if not pack_opts:
        print(f"{_LOG} Pretrained data.dat has no 'options' key — skipping.", flush=True)
        return None

    problems = _validate_options(pack_opts)
    if problems:
        print(f"{_LOG} Pretrained data.dat failed validation — skipping:", flush=True)
        for p in problems:
            print(f"{_LOG}   {p}", flush=True)
        return None

    # ── 3. Determine expected weight files ────────────────────────────────────
    archi = pack_opts.get("archi", "df")
    needed = _weight_files(archi)

    # ── 4. Confirm weight files are present ───────────────────────────────────
    missing = [f for f in needed if not (pretrained_dir / f"{prefix}{f}").exists()]
    if missing:
        print(f"{_LOG} Pretrained pack is missing weight files: {missing} — skipping.",
              flush=True)
        return None

    # ── 5. Copy weights into model_dir with session name prefix ───────────────
    res        = pack_opts.get("resolution", "?")
    pack_iter  = pack_data.get("iter", 0)

    if prefix:
        print(f"{_LOG} Pretrained pack found: {pretrained_dir} (file prefix: '{prefix}')",
              flush=True)
    else:
        print(f"{_LOG} Pretrained pack found: {pretrained_dir}", flush=True)
    print(f"{_LOG}   resolution={res}  archi={archi}  "
          f"ae_dims={pack_opts.get('ae_dims')}  "
          f"e_dims={pack_opts.get('e_dims')}  "
          f"d_dims={pack_opts.get('d_dims')}",
          flush=True)
    print(f"{_LOG}   Pack trained for {pack_iter:,} iterations.", flush=True)

    for fname in needed:
        src = pretrained_dir / f"{prefix}{fname}"
        dst = model_dir / f"{model_name}_{fname}"
        shutil.copy2(src, dst)
        print(f"{_LOG}   copied  {prefix}{fname}  ->  {dst.name}", flush=True)

    print(f"{_LOG} Bootstrap complete — session will warm-start from pretrained weights.",
          flush=True)
    return pack_opts


# ── Internal helpers ──────────────────────────────────────────────────────────

def _weight_files(archi: str) -> tuple[str, ...]:
    """Return the weight filenames expected for this archi string."""
    archi_type = archi.split("-")[0]
    return _LIAE_WEIGHTS if archi_type == "liae" else _DF_WEIGHTS


def _is_int_like(val) -> bool:
    """Return True for Python int or any numpy integer type."""
    if isinstance(val, int):
        return True
    # Accept numpy integer scalars without importing numpy
    t = type(val)
    return issubclass(t, int) or getattr(t, "__mro__", None) and any(
        c.__name__ in ("integer", "signedinteger", "unsignedinteger")
        for c in t.__mro__
    )


def _validate_options(opts: dict) -> list[str]:
    """
    Basic sanity check on options read from a pretrained data.dat.
    Returns a list of problem strings; empty means OK.
    Accepts both Python int and numpy integer types (packs from DFL use numpy.int64).
    """
    problems: list[str] = []

    res = opts.get("resolution")
    if not _is_int_like(res) or not (64 <= res <= 640):
        problems.append(
            f"resolution={res!r} is invalid (expected int 64–640)"
        )

    archi = opts.get("archi", "")
    archi_type = archi.split("-")[0] if isinstance(archi, str) else ""
    if archi_type not in _VALID_ARCHI_TYPES:
        problems.append(
            f"archi={archi!r} is not recognised "
            f"(expected a 'df' or 'liae' variant)"
        )

    for dim_key in ("ae_dims", "e_dims", "d_dims", "d_mask_dims"):
        val = opts.get(dim_key)
        if val is not None and (not _is_int_like(val) or val < 4):
            problems.append(f"{dim_key}={val!r} is invalid (expected int >= 4)")

    return problems
