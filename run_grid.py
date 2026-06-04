"""
run_grid.py
===========
Run a grid of FedFRESH simulations. The grid is an explicit LIST OF RUNS, so you
control exactly which simulations execute -- no Cartesian-product redundancy.

Why a list and not a product: the `temp` argument means different things per
mode --
    * async_asymp_EI   : temp = alpha   (alpha-fair parameter)
    * fedstale         : temp = beta    (FedStale memory coefficient, in [0,1])
    * async_asymp_age  : temp UNUSED
    * fedbuff          : temp UNUSED
So sweeping temp across all modes would run age/fedbuff redundantly. The list
below lets EI take a temp sweep while each baseline runs exactly once.

How it works:
    * SHARED  -- parameters applied to EVERY run (e.g. user_prob_disc, cuda).
    * RUNS    -- one dict per simulation; its keys override SHARED.
    * Each run's parameters = SHARED merged with the run dict.
    * The runner passes ONLY the keys you list; every other setting comes from
      the defaults in lr_001_5class_CustomCNN_MNIST_batuFlavor.py and is
      therefore identical across all runs.

Each run's CSV outputs are written to  grid_results/<label>/  by the simulator
when that run completes. A manifest is (re)written after every run, so an
interrupted grid still leaves a correct record of what finished.

Usage:
    python3 run_grid.py                 # run the whole grid
    python3 run_grid.py --dry-run       # print the commands, run nothing
    python3 run_grid.py --cuda 2        # override the CUDA device for all runs

NOTE: each run is a full FL training and can take a long time. Use --dry-run
first to confirm the grid.
"""

import argparse
import os
import subprocess
import sys
import time

# ===========================================================================
#  SET YOUR GRID HERE
# ===========================================================================
# Parameters applied to every run. Put the single CUDA device here.
SHARED = {
    "user_prob_disc": 0.45,
    "cuda": 0,
}

# One dict per simulation. Keys here override SHARED. Omit `temp` for modes
# that do not use it (age, fedbuff).
RUNS = [
    {"selected_mode": "async_asymp_EI", "temp": 0},      # alpha = 0   (greedy)
    {"selected_mode": "async_asymp_EI", "temp": 0.25},   # alpha = 0.25
    {"selected_mode": "async_asymp_EI", "temp": 0.5},    # alpha = 0.5
    {"selected_mode": "async_asymp_EI", "temp": "inf"},  # alpha = inf (equal impact)
    {"selected_mode": "async_asymp_age"},                # baseline, temp unused
    {"selected_mode": "fedbuff"},                        # baseline, temp unused
    {"selected_mode": "fedstale", "temp": 0},          # temp = beta for fedstale
    {"selected_mode": "fedstale", "temp": 0.5},          # temp = beta for fedstale
    {"selected_mode": "fedstale", "temp": 1},          # temp = beta for fedstale
]

GRID_OUT = "grid_results"   # parent folder for all run outputs (relative to this file)
# ===========================================================================

SIM = "lr_001_5class_CustomCNN_MNIST_batuFlavor.py"

# lr_001 arguments the runner is allowed to set (--out_dir is set automatically).
VALID_ARGS = {
    "learning_rate_client", "learning_rate_server", "epochs", "batch_size",
    "num_users", "fraction", "num_timeframes", "seeds", "num_runs",
    "selected_mode", "cos_similarity", "train_mode", "bufferLimit",
    "theta_inner", "data_mode", "unit_gradients", "adam", "temp",
    "cos_similarity_type", "user_prob_disc", "cuda",
}


def fmt(v):
    """Filesystem-safe string for a parameter value."""
    return str(v).replace(" ", "").replace("/", "-")


def cmd_args(params):
    """Flatten a {key: value} dict into ['--key', 'value', ...] for argparse."""
    out = []
    for k, v in params.items():
        out.append(f"--{k}")
        if isinstance(v, (list, tuple)):       # nargs='+' arguments (e.g. seeds)
            out.extend(str(x) for x in v)
        else:
            out.append(str(v))
    return out


def write_manifest(path, results):
    with open(path, "w") as f:
        f.write("idx\tlabel\tselected_mode\ttemp\tuser_prob_disc\tcuda\tstatus\tout_dir\n")
        for r in results:
            f.write(f"{r['idx']}\t{r['label']}\t{r['mode']}\t{r['temp']}\t"
                    f"{r['disc']}\t{r['cuda']}\t{r['status']}\t{r['out_dir']}\n")


def main():
    ap = argparse.ArgumentParser(description="Grid runner for FedFRESH simulations.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands without executing them.")
    ap.add_argument("--python", default=sys.executable or "python3",
                    help="Python interpreter to use for the simulations.")
    ap.add_argument("--cuda", type=int, default=None,
                    help="Override the CUDA device for ALL runs (else SHARED['cuda']).")
    args = ap.parse_args()

    sim_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(sim_dir, SIM)
    if not os.path.isfile(sim_path):
        sys.exit(f"ERROR: simulator not found: {sim_path}")

    # ---- validate the grid ------------------------------------------------
    if not RUNS:
        sys.exit("ERROR: RUNS is empty.")
    for j, run in enumerate(RUNS):
        if "selected_mode" not in run:
            sys.exit(f"ERROR: RUNS[{j}] is missing 'selected_mode'.")
        for k in list(run) + list(SHARED):
            if k not in VALID_ARGS:
                sys.exit(f"ERROR: unknown parameter '{k}' "
                         f"(not an argument of {SIM}).")

    grid_out = os.path.join(sim_dir, GRID_OUT)

    # one CUDA device for the whole grid: CLI flag overrides SHARED
    cuda_dev = args.cuda if args.cuda is not None else SHARED.get("cuda")

    print(f"Grid: {len(RUNS)} run(s)  ->  {grid_out}")
    print(f"  CUDA device (all runs): {cuda_dev}")
    print(f"  shared parameters     : "
          f"{ {k: v for k, v in SHARED.items() if k != 'cuda'} }")
    print(f"  (every other setting fixed at the defaults in {SIM})\n")

    results = []
    manifest_path = os.path.join(grid_out, "manifest.txt")

    for i, run in enumerate(RUNS):
        params = dict(SHARED)
        params.update(run)
        if cuda_dev is not None:
            params["cuda"] = cuda_dev

        mode = params["selected_mode"]
        temp = params.get("temp", "(default)")
        label = f"{i:02d}_{fmt(mode)}"
        if "temp" in params:
            label += f"_t{fmt(temp)}"
        out_dir = os.path.join(grid_out, label)

        cmd = [args.python, sim_path] + cmd_args(params) + ["--out_dir", out_dir]

        print(f"[{i + 1}/{len(RUNS)}] {label}")
        print("  " + " ".join(cmd))

        rec = dict(idx=i, label=label, mode=mode, temp=temp,
                   disc=params.get("user_prob_disc", "(default)"),
                   cuda=params.get("cuda", "(default)"),
                   out_dir=out_dir, status="dry-run")

        if not args.dry_run:
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=sim_dir)   # stdout/stderr stream live
            dt = time.time() - t0
            rec["status"] = ("OK" if proc.returncode == 0
                             else f"FAILED(rc={proc.returncode})")
            print(f"  -> {rec['status']}  ({dt:.0f}s)\n")

        results.append(rec)

        # (re)write the manifest after every run so progress is never lost
        if not args.dry_run:
            os.makedirs(grid_out, exist_ok=True)
            write_manifest(manifest_path, results)

    # ---- summary ----------------------------------------------------------
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        print(f"  {r['status']:>16}  {r['label']}")
    if not args.dry_run:
        print(f"\nManifest written to: {manifest_path}")

    failed = [r for r in results if r["status"].startswith("FAILED")]
    if failed:
        sys.exit(f"\n{len(failed)} of {len(results)} run(s) FAILED.")


if __name__ == "__main__":
    main()
