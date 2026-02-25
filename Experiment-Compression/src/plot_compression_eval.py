import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Resolve ROOT relative to the repository root (two levels up from this script)
ROOT = Path(__file__).resolve().parent.parent.parent / "results" / "compression_eval"


# Distinct, colorblind-friendly palette (Tableau 10 + black for baseline)
COLORS: Dict[str, str] = {
    # Baseline
    "Baseline": "#000000",  # Black

    # Pooling methods
    "Hierarchical Pooling": "#4E79A7",  # Blue
    "Spherical Pooling": "#59A14F",    # Green
    "Spherical Pooling+": "#1F8B4C",   # Dark Green
    "IDF Pooling": "#F28E2B",         # Orange
    "Random pooling": "#B07AA1",       # Purple
    "Attention score pooling": "#E15759",  # Red

    # Pruning methods
    "Random pruning": "#76B7B2",       # Teal
    "Attention score pruning": "#EDC948",  # Yellow
    "Leverage score pruning": "#9C755F",   # Brown
    "Doc-wise IDF pruning": "#BAB0AC",    # Gray
}


MARKERS: Dict[str, str] = {
    "Baseline": "*",
    "Hierarchical Pooling": "o",
    "Spherical Pooling": "s",
    "Spherical Pooling+": "D",
    "IDF Pooling": "^",
    "Random pooling": "v",
    "Attention score pooling": "<",
    "Random pruning": "8",
    "Attention score pruning": ">",
    "Leverage score pruning": "D",
    "Doc-wise IDF pruning": "P",
}


KNOWN_METHODS: List[str] = [
    "Baseline",
    "Hierarchical Pooling",
    "Spherical Pooling+",
    "Spherical Pooling",
    "IDF Pooling",
    "Random pooling",
    "Attention score pooling",
    "Random pruning",
    "Attention score pruning",
    "Leverage score pruning",
    "Doc-wise IDF pruning",
]


def infer_method(config_name: str) -> str:
    if config_name == "Baseline":
        return "Baseline"
    for m in KNOWN_METHODS:
        if config_name.startswith(m):
            return m
    # Fallback: first two words
    return " ".join(config_name.split()[:2])


def pretty_dataset_name(dataset_id: str) -> str:
    """Return a short human-readable name for a dataset.

    Absolute filesystem paths are shortened to their last two components
    (e.g. ``amazon_dataset/beir_format_full``).  ir_datasets-style IDs such
    as ``beir/nfcorpus/test`` are returned unchanged.
    """
    p = Path(dataset_id)
    if p.is_absolute():
        # Use at most the last two path components
        parts = p.parts
        return "/".join(parts[-2:]) if len(parts) >= 2 else p.name
    return dataset_id


def load_run(run_dir: Path) -> pd.DataFrame:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return pd.DataFrame()

    rows = []
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            eval_scores = data.get("evaluation", {})
            rows.append({
                "config_idx": data.get("config_idx"),
                "config_name": data.get("config_name"),
                "method": infer_method(data.get("config_name", "")),
                "token_count": data.get("token_count"),
                "avg_tokens_per_doc": data.get("avg_tokens_per_doc"),
                **eval_scores,
            })
    df = pd.DataFrame(rows)
    # Guard against results from a second evaluation being appended to the
    # same file.  Detect the boundary by finding the first row whose
    # config_idx already appeared earlier (i.e. the second evaluation's
    # configs start repeating indices from the first evaluation).
    if "config_idx" in df.columns:
        seen_idx: set = set()
        split_at = len(df)
        for i, cidx in enumerate(df["config_idx"]):
            if cidx in seen_idx:
                split_at = i
                break
            seen_idx.add(cidx)
        if split_at < len(df):
            df = df.iloc[:split_at].reset_index(drop=True)
    # Drop duplicate config_names (resume runs may re-evaluate existing
    # configs).  Keep the last occurrence (most recent evaluation).
    if "config_name" in df.columns:
        df = df.drop_duplicates(subset="config_name", keep="last").reset_index(drop=True)
    # Attach metadata if present
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            df.attrs["dataset_id"] = meta.get("dataset_id")
            df.attrs["model_name"] = meta.get("model_name")
            df.attrs["run_id"] = meta.get("run_id")
            df.attrs["metrics"] = meta.get("config", {}).get("metrics", [])
        except Exception:
            pass
    return df


def plot_run(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Determine metrics to plot
    default_metrics = ["ndcg@10", "map", "recall@10", "mrr@10", "precision@10"]
    metrics = [m for m in default_metrics if m in df.columns]
    title_suffix = ""
    if df.attrs.get("dataset_id"):
        title_suffix += f" | {pretty_dataset_name(df.attrs['dataset_id'])}"
    if df.attrs.get("model_name"):
        title_suffix += f" | {df.attrs['model_name']}"

    # Exclude requested methods (e.g., Leverage score pruning)
    if "method" in df.columns:
        df = df[df["method"].str.lower() != "leverage score pruning".lower()]

    # Sort once for consistency
    df = df.sort_values(by=["method", "avg_tokens_per_doc"])  # type: ignore[arg-type]

    # Compute baseline tokens once (if available)
    baseline_tokens = None
    if "method" in df.columns and "avg_tokens_per_doc" in df.columns:
        base_df = df[df["method"] == "Baseline"]
        if not base_df.empty:
            try:
                baseline_tokens = float(base_df.iloc[0]["avg_tokens_per_doc"])  # type: ignore[index]
            except Exception:
                baseline_tokens = None

    for metric in metrics:
        plt.figure(figsize=(9, 6))
        ax = plt.gca()

        for method, group in df.groupby("method"):
            xs = group["avg_tokens_per_doc"].values
            ys = group[metric].values
            marker = MARKERS.get(method, "o")
            color = COLORS.get(method, "#333333")
            ax.plot(
                xs, ys,
                marker=marker,
                linewidth=1.25,
                markersize=8,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                alpha=0.9,
                label=method,
            )

        ax.set_xlabel("Average tokens per document")
        ax.set_ylabel(metric)
        ax.set_title(f"Compression comparison{title_suffix}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        # Reference lines and axis limits starting from 0 to 80% of baseline tokens
        if baseline_tokens is not None:
            fifty_tokens = 0.5 * baseline_tokens
            eighty_tokens = 0.8 * baseline_tokens
            # Vertical dashed line at 50% compression
            ax.axvline(
                x=fifty_tokens,
                linestyle="--",
                linewidth=1.0,
                color="#666666",
                alpha=0.85,
                label="_nolegend_",
            )
            # Horizontal dashed line at baseline performance for this metric (if available)
            base_df_metric = df[df["method"] == "Baseline"]
            if not base_df_metric.empty and metric in base_df_metric.columns:
                try:
                    baseline_metric_value = float(base_df_metric.iloc[0][metric])  # type: ignore[index]
                    ax.axhline(
                        y=baseline_metric_value,
                        linestyle="--",
                        linewidth=1.0,
                        color="#999999",
                        alpha=0.85,
                        label="_nolegend_",
                    )
                except Exception:
                    pass
            # Set x-axis from 0 up to 80% of baseline tokens
            ax.set_xlim(left=0.0, right=eighty_tokens)
            # Add annotations: '50%' near x-axis and 'baseline' near top-right
            try:
                x_left, x_right = ax.get_xlim()
                y_bottom, y_top = ax.get_ylim()
                # Label for 50% vertical line near the x-axis
                ax.text(
                    fifty_tokens,
                    y_bottom + 0.02 * (y_top - y_bottom),
                    "50%",
                    color="#666666",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                )
                # Label for baseline horizontal line near the top-right corner
                if 'baseline_metric_value' in locals():
                    ax.text(
                        x_right * 0.99,
                        baseline_metric_value,
                        "baseline",
                        color="#999999",
                        fontsize=9,
                        ha="right",
                        va="center",
                    )
            except Exception:
                pass
        # One legend entry per method
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=9)
        plt.tight_layout()
        out_path = out_dir / f"compression_eval_{metric.replace('@', 'at')}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and sort a DataFrame for plotting (shared logic)."""
    if "method" in df.columns:
        df = df[df["method"].str.lower() != "leverage score pruning".lower()]
    df = df.sort_values(by=["method", "avg_tokens_per_doc"])
    return df


def _baseline_tokens(df: pd.DataFrame) -> Optional[float]:
    if "method" in df.columns and "avg_tokens_per_doc" in df.columns:
        base = df[df["method"] == "Baseline"]
        if not base.empty:
            try:
                return float(base.iloc[0]["avg_tokens_per_doc"])
            except Exception:
                pass
    return None


def _add_reference_lines(ax, df: pd.DataFrame, metric: str, baseline_tokens: Optional[float]) -> None:
    """Draw baseline / 50 % reference lines on *ax*."""
    if baseline_tokens is None:
        return
    fifty = 0.5 * baseline_tokens
    eighty = 0.8 * baseline_tokens
    ax.axvline(x=fifty, linestyle="--", linewidth=1.0, color="#666666", alpha=0.85, label="_nolegend_")
    base_rows = df[df["method"] == "Baseline"]
    baseline_metric_value = None
    if not base_rows.empty and metric in base_rows.columns:
        try:
            baseline_metric_value = float(base_rows.iloc[0][metric])
            ax.axhline(y=baseline_metric_value, linestyle="--", linewidth=1.0, color="#999999", alpha=0.85, label="_nolegend_")
        except Exception:
            pass
    ax.set_xlim(left=0.0, right=eighty)
    try:
        _, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        ax.text(fifty, y_bottom + 0.02 * (y_top - y_bottom), "50%", color="#666666", fontsize=9, ha="center", va="bottom")
        if baseline_metric_value is not None:
            ax.text(x_right * 0.99, baseline_metric_value, "baseline", color="#999999", fontsize=9, ha="right", va="center")
    except Exception:
        pass


def _plot_methods_on_ax(ax, df: pd.DataFrame, metric: str) -> None:
    """Plot each method's curve on *ax*."""
    for method, group in df.groupby("method"):
        xs = group["avg_tokens_per_doc"].values
        ys = group[metric].values
        ax.plot(
            xs, ys,
            marker=MARKERS.get(method, "o"),
            linewidth=1.25,
            markersize=8,
            color=COLORS.get(method, "#333333"),
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=0.9,
            label=method,
        )


# ── Combined subplot plot (one panel per dataset) ────────────────────────────

def plot_combined(
    run_dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """For each metric produce a single figure with one subplot per dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)

    default_metrics = ["ndcg@10", "map", "recall@10", "mrr@10", "precision@10"]
    # Only keep metrics present in *all* DataFrames
    metrics = [m for m in default_metrics if all(m in df.columns for df in run_dfs.values())]
    if not metrics:
        print("  No common metrics across runs – skipping combined plots.")
        return

    n_runs = len(run_dfs)
    for metric in metrics:
        fig, axes = plt.subplots(1, n_runs, figsize=(7 * n_runs, 6), squeeze=False)
        for idx, (label, raw_df) in enumerate(run_dfs.items()):
            ax = axes[0, idx]
            df = _prepare_df(raw_df)
            bt = _baseline_tokens(df)
            _plot_methods_on_ax(ax, df, metric)
            _add_reference_lines(ax, df, metric, bt)
            ax.set_xlabel("Average tokens per document")
            ax.set_ylabel(metric)
            ax.set_title(label, fontsize=11)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
            # De-duplicate legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=8)

        fig.suptitle(f"Compression comparison – {metric}", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        out_path = out_dir / f"combined_{metric.replace('@', 'at')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved combined plot: {out_path}")


# ── Average plot (mean across datasets) ──────────────────────────────────────

def plot_average(
    run_dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """For each metric produce a single plot with scores averaged across datasets."""
    out_dir.mkdir(parents=True, exist_ok=True)

    default_metrics = ["ndcg@10", "map", "recall@10", "mrr@10", "precision@10"]
    metrics = [m for m in default_metrics if all(m in df.columns for df in run_dfs.values())]
    if not metrics:
        print("  No common metrics across runs – skipping average plots.")
        return

    # Prepare all DataFrames
    prepped: List[pd.DataFrame] = []
    for raw_df in run_dfs.values():
        prepped.append(_prepare_df(raw_df))

    # Concatenate and average by (method, config_name)
    combined = pd.concat(prepped, ignore_index=True)
    # Group by method + config_name so each (method, compression-level) pair is averaged
    group_cols = ["method", "config_name"]
    agg_cols = ["avg_tokens_per_doc"] + metrics
    avg_df = combined.groupby(group_cols, as_index=False)[agg_cols].mean()
    avg_df = avg_df.sort_values(by=["method", "avg_tokens_per_doc"])

    bt = _baseline_tokens(avg_df)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 6))
        _plot_methods_on_ax(ax, avg_df, metric)
        _add_reference_lines(ax, avg_df, metric, bt)
        ax.set_xlabel("Average tokens per document")
        ax.set_ylabel(metric)
        ax.set_title(f"Compression comparison – average across {len(run_dfs)} datasets")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=9)
        fig.tight_layout()
        out_path = out_dir / f"average_{metric.replace('@', 'at')}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Saved average plot: {out_path}")


# ── CSV export ────────────────────────────────────────────────────────────────

def _save_results_csvs(
    run_dfs: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Save per-dataset and averaged results as CSVs into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    default_metrics = ["ndcg@10", "map", "recall@10", "mrr@10", "precision@10"]
    metrics = [m for m in default_metrics if all(m in df.columns for df in run_dfs.values())]

    # Per-dataset CSVs + one combined CSV with a "dataset" column
    all_parts: List[pd.DataFrame] = []
    for label, raw_df in run_dfs.items():
        df = _prepare_df(raw_df)
        cols = [c for c in ["method", "config_name", "avg_tokens_per_doc"] + metrics if c in df.columns]
        part = df[cols].copy()
        part.insert(0, "dataset", label)
        all_parts.append(part)

    if all_parts:
        combined_csv = pd.concat(all_parts, ignore_index=True)
        combined_csv = combined_csv.sort_values(by=["dataset", "method", "avg_tokens_per_doc"])
        p = out_dir / "results_per_dataset.csv"
        combined_csv.to_csv(p, index=False)
        print(f"  Saved per-dataset CSV: {p}")

    # Averaged CSV
    prepped = [_prepare_df(raw_df) for raw_df in run_dfs.values()]
    combined = pd.concat(prepped, ignore_index=True)
    group_cols = ["method", "config_name"]
    agg_cols = ["avg_tokens_per_doc"] + metrics
    agg_cols = [c for c in agg_cols if c in combined.columns]
    avg_df = combined.groupby(group_cols, as_index=False)[agg_cols].mean()
    avg_df = avg_df.sort_values(by=["method", "avg_tokens_per_doc"])
    p = out_dir / "results_average.csv"
    avg_df.to_csv(p, index=False)
    print(f"  Saved average CSV: {p}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Plot compression eval results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to results directory (default: results/compression_eval next to repo root).",
    )
    args, _ = parser.parse_known_args()

    results_dir = args.results_dir if args.results_dir is not None else ROOT
    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        return

    # Iterate timestamped runs (directories containing results.jsonl)
    run_dirs = [p for p in results_dir.iterdir() if p.is_dir() and (p / "results.jsonl").exists()]
    if not run_dirs:
        print(f"No runs with results.jsonl found under {results_dir}")
        return

    # ── Per-run plots (existing behaviour) ──
    run_dfs: Dict[str, pd.DataFrame] = {}
    for run_dir in sorted(run_dirs):
        print(f"Processing: {run_dir}")
        df = load_run(run_dir)
        if df.empty:
            print(f"  Skipping (no rows): {run_dir}")
            continue

        # Save summary TSV
        summary_path = run_dir / "summary.tsv"
        df.to_csv(summary_path, sep="\t", index=False)

        # Per-run plots
        plots_dir = run_dir / "plots"
        plot_run(df, plots_dir)
        print(f"  Saved per-run plots to: {plots_dir}")

        # Build label for combined / average plots (disambiguate collisions)
        raw_label = df.attrs.get("dataset_id") or run_dir.name
        label = pretty_dataset_name(raw_label)
        if label in run_dfs:
            label = f"{label} ({run_dir.name})"
        run_dfs[label] = df

    # ── Combined + average plots (all runs together) ──
    if len(run_dfs) >= 2:
        combined_dir = results_dir / "combined_plots"
        print(f"\nGenerating combined subplot plots …")
        plot_combined(run_dfs, combined_dir)
        print(f"Generating average plots …")
        plot_average(run_dfs, combined_dir)

        # ── Dump CSVs into combined_plots ──
        _save_results_csvs(run_dfs, combined_dir)
    else:
        print("Only one run found – skipping combined / average plots.")


if __name__ == "__main__":
    main()
