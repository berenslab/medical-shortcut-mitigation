import argparse
import glob
import os
import re

import pandas as pd


def aggregate_metrics(
    root_dir: str,
    version: str,
    mode: str = "val",
    output_filename_prefix: str = None,
    sort_metrics: bool = True,
):
    """Aggregates metric CSVs from all folds under a specific version directory.

    Args:
        root_dir: Root directory containing `fold*/lightning_logs/version_x/` folders.
        version: The version subdirectory to use (e.g., "version_0").
        mode: Model mode, one of ["val", "test_inverse", "test_balanced"].
        output_filename_prefix: Prefix for output files.
        sort_metrics: If True, the metrics are ordered alphabetically.
    """
    pattern = os.path.join(
        root_dir,
        "fold*/lightning_logs",
        f"version_{version}",
        f"{mode}_metrics_fold_*.csv",
    )
    files = glob.glob(pattern)

    if not files:
        print(f"[!] No {mode} metric files found for version '{version}' in {root_dir}")
        return

    print(f"[✓] Found {len(files)} {mode} metric files for version {version}")

    save_dir = os.path.join(root_dir, f"eval-version{version}")
    os.makedirs(save_dir, exist_ok=True)

    all_metrics = []
    for filepath in files:
        df = pd.read_csv(filepath)

        # Extract fold number
        if "fold" in df['metric'].values:
            fold = int(df[df['metric'] == 'fold']['value'].values[0])
        else:
            match = re.search(r"fold(\d+)", filepath)
            fold = int(match.group(1)) if match else -1

        # Remove outlier rows (fold, step, epoch)
        df = df[~df['metric'].isin(['fold', 'step', 'epoch', 'stage'])]

        # Pivot metrics to columns
        df_wide = df.pivot_table(index=None, columns='metric', values='value', aggfunc='first')
        df_wide["fold"] = fold
        all_metrics.append(df_wide)

    full_df = pd.concat(all_metrics, ignore_index=True)

    # Sort metric columns alphabetically (except 'fold')
    metric_cols = [col for col in full_df.columns if col != 'fold']
    if sort_metrics:
        metric_cols.sort()

    full_df = full_df[metric_cols + ["fold"]]

    # Save raw metrics
    raw_filename = f"{output_filename_prefix or mode}_metrics_raw_{version}.csv"
    raw_path = os.path.join(save_dir, raw_filename)
    full_df.to_csv(raw_path, index=False)
    print(f"[✓] Saved raw per-fold metrics to: {raw_path}")

    # Summary
    summary_df = pd.DataFrame({
        "metric": metric_cols,
        "mean": full_df[metric_cols].astype(float).mean().values,
        "std": full_df[metric_cols].astype(float).std().values
    }).round(3)

    summary_filename = f"{output_filename_prefix or mode}_metrics_summary_{version}.csv"
    summary_path = os.path.join(save_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)
    print(f"[✓] Saved metric summary to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate fold metrics for a specific version"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory with fold*/lightning_logs/version_x/",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0,
        help="The version_x directory name to use for aggregation",
    )
    parser.add_argument(
        "--modes",
        type=list,
        default=["val", "test_original", "test_inverse", "test_balanced"],
        help="Which type of metrics to aggregate",
    )
    parser.add_argument(
        "--output_filename_prefix",
        type=str,
        default=None,
        help="Optional prefix for output filenames",
    )

    args = parser.parse_args()
    for mode in args.modes:
        aggregate_metrics(
            root_dir=args.root_dir,
            version=str(args.version),
            mode=mode,
            output_filename_prefix=args.output_filename_prefix,
        )
