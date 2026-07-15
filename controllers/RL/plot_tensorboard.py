"""
Analyze and plot TensorBoard logs produced during PPO training.

Usage:
    python plot_tensorboard.py                       # plot latest run
    python plot_tensorboard.py --run PPO_3            # plot a specific run
    python plot_tensorboard.py --list                 # list available runs
    python plot_tensorboard.py --compare               # overlay all runs
    python plot_tensorboard.py --output ../reports/run3.png

"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import TENSORBOARD_LOG_DIR

# Metrics SB3 logs by default for PPO that are useful to track
DEFAULT_TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/explained_variance",
    "train/approx_kl",
    "train/clip_fraction",
]


def list_runs(logdir: str) -> List[Path]:
    """List available training run directories (e.g. PPO_1, PPO_2, ...)."""
    root = Path(logdir)
    if not root.exists():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )


def load_scalars(run_dir: Path) -> Dict[str, List[tuple]]:
    """
    Load all scalar tags from a run directory.

    Returns:
        Dict mapping tag name -> list of (step, value) tuples
    """
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()

    available_tags = accumulator.Tags().get("scalars", [])
    data = {}
    for tag in available_tags:
        events = accumulator.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data


def plot_run(run_dir: Path, tags: Optional[List[str]] = None,
             output: Optional[str] = None) -> None:
    """Plot scalar metrics for a single run in a grid of subplots."""
    print(f"\n Loading run: {run_dir.name}")
    data = load_scalars(run_dir)

    if not data:
        print(f" No scalar data found in: {run_dir}")
        return

    tags_to_plot = [t for t in (tags or DEFAULT_TAGS) if t in data]
    missing = [t for t in (tags or DEFAULT_TAGS) if t not in data]
    if missing:
        print(f"  Not found in this run (skipped): {', '.join(missing)}")

    if not tags_to_plot:
        print(" None of the requested tags exist in this run. Available tags:")
        for t in sorted(data.keys()):
            print(f"    - {t}")
        return

    n = len(tags_to_plot)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, tag in zip(axes, tags_to_plot):
        steps, values = zip(*data[tag])
        ax.plot(steps, values, linewidth=1.5)
        ax.set_title(tag)
        ax.set_xlabel("timesteps")
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Training run: {run_dir.name}", fontsize=14)
    fig.tight_layout()

    _save_or_show(fig, output, default_name=f"{run_dir.name}.png")


def plot_comparison(run_dirs: List[Path], tag: str = "rollout/ep_rew_mean",
                     output: Optional[str] = None) -> None:
    """Overlay a single metric across multiple runs for comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_any = False
    for run_dir in run_dirs:
        data = load_scalars(run_dir)
        if tag not in data:
            print(f"  '{tag}' not found in {run_dir.name}, skipping")
            continue
        steps, values = zip(*data[tag])
        ax.plot(steps, values, label=run_dir.name, linewidth=1.5)
        plotted_any = True

    if not plotted_any:
        print(f" No runs contained tag '{tag}'")
        plt.close(fig)
        return

    ax.set_title(f"Comparison: {tag}")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(tag)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, output, default_name="comparison.png")


def _save_or_show(fig, output: Optional[str], default_name: str) -> None:
    """Save figure to disk if output path given, otherwise display it."""
    if output:
        out_path = Path(output)
    else:
        out_path = Path(TENSORBOARD_LOG_DIR).parent / "plots" / default_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f" Plot saved to: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PPO TensorBoard training logs")
    parser.add_argument(
        "--logdir",
        type=str,
        default=TENSORBOARD_LOG_DIR,
        help=f"TensorBoard log directory (default: {TENSORBOARD_LOG_DIR})",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run name to plot (e.g. PPO_3). Default: most recent run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available runs and exit",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Overlay a single metric across all runs instead of plotting one run in detail",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="rollout/ep_rew_mean",
        help="Metric tag to use with --compare (default: rollout/ep_rew_mean)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot image (default: <logdir>/../plots/<name>.png)",
    )

    args = parser.parse_args()

    runs = list_runs(args.logdir)
    if not runs:
        print(f"ERROR: No runs found in log directory: {args.logdir}")
        return

    if args.list:
        print(f"\nAvailable runs in {args.logdir}:")
        for r in runs:
            print(f"  - {r.name}")
        return

    if args.compare:
        plot_comparison(runs, tag=args.tag, output=args.output)
        return

    if args.run:
        matches = [r for r in runs if r.name == args.run]
        if not matches:
            print(f"ERROR: Run '{args.run}' not found. Available runs:")
            for r in runs:
                print(f"  - {r.name}")
            return
        target_run = matches[0]
    else:
        target_run = runs[-1]  # most recently modified

    plot_run(target_run, output=args.output)


if __name__ == "__main__":
    main()