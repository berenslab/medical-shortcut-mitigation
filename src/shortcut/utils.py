import os
import string
import time
from os import path as osp
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from lightning.pytorch.callbacks import (Callback, ModelCheckpoint,
                                         RichProgressBar)
from lightning.pytorch.utilities.rank_zero import (rank_zero_info,
                                                   rank_zero_only)
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties


class LitProgressBar(RichProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def get_metrics(self, trainer, pl_module):
        # Don't show the version number.
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def save_torchscript(module: torch.nn.Module, filepath: str) -> None:
    scripted_model = torch.jit.script(module)
    scripted_model.save(filepath)


class TorchScriptModelCheckpoint(ModelCheckpoint):
    r"""Saves the model as an additional standalone pt file whenever a checkpoint is created."""

    def __init__(
        self,
        dirpath=None,
        filename=None,
        monitor=None,
        verbose=False,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
        auto_insert_metric_name=True,
        every_n_train_steps=None,
        train_time_interval=None,
        every_n_epochs=None,
        save_on_train_epoch_end=None,
        enable_version_counter=True,
    ):
        super(TorchScriptModelCheckpoint, self).__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        self.last_kth_best_model_path = ""

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> dict:
        """
        Convert model to TorchScript and save it as a .pt file
        after training ends (or at any checkpoint saving step).
        """
        if not osp.exists(self.dirpath):
            os.mkdir(self.dirpath)
        callback_metrics = {
            key: int(val) if key == "step" else val
            for key, val in trainer.callback_metrics.items()
        }
        callback_metrics["epoch"] = trainer.current_epoch
        filename, _ = os.path.splitext(
            self.format_checkpoint_name(callback_metrics, self.filename)
        )
        torchscript_model_path = f"{filename}.pt"

        # Save the model
        save_torchscript(pl_module, torchscript_model_path)

        # Optionally, you can include the TorchScript model path in the checkpoint (if you want)
        checkpoint["torchscript_model_path"] = torchscript_model_path

        # Remove (k+1)th model
        if self.last_kth_best_model_path != "":
            filename_to_delete, _ = os.path.splitext(self.last_kth_best_model_path)
            os.remove(f"{filename_to_delete}.pt")

        # Update deadthlist
        self.last_kth_best_model_path = self.kth_best_model_path

        return checkpoint


class DelayedCheckpoint(ModelCheckpoint):
    def __init__(self, start_saving_epoch=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_saving_epoch = start_saving_epoch
        self._reset_done = False  # track whether best_model_score has been reset

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if current_epoch >= self.start_saving_epoch:
            # Reset best score exactly once when saving starts
            if not self._reset_done:
                if self.mode == "min":
                    self.best_model_score = float("inf")
                else:
                    self.best_model_score = float("-inf")
                self.best_k_models = {}
                self.kth_best_model_path = ""
                self.kth_value = None
                self.best_model_path = ""
                self._reset_done = True

            # Call the original ModelCheckpoint logic
            super().on_train_epoch_end(trainer, pl_module)


class BestValMetricsLoggerCallback(Callback):
    """Tracks best validation metrics internally.

    Attributes:
        fold: Current fold number (used in CSV filename if enabled).
        start_saving_epoch: Epoch number to start collecting metrics from.
        save_csv: Whether to save best metrics to a CSV at the end.
    """

    def __init__(self, fold: int, start_saving_epoch: int = 0, save_csv: bool = True):
        self.fold = fold
        self.start_saving_epoch = start_saving_epoch
        self.save_csv = save_csv
        self.val_metrics = []

    def setup(self, trainer, pl_module, stage=None):
        if self.save_csv:
            base_dir = trainer.logger.log_dir
            self.val_csv_path = os.path.join(
                base_dir, f"val_metrics_fold_{self.fold}.csv"
            )

    def _extract_val_metrics(self, metrics):
        # Only take metrics starting with 'val' and convert tensors to Python scalars.
        return {
            k: v.item() if hasattr(v, "item") else v
            for k, v in metrics.items()
            if k.startswith("val")
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        if (not trainer.sanity_checking) and (
            trainer.current_epoch >= self.start_saving_epoch
        ):
            val_metrics = self._extract_val_metrics(trainer.callback_metrics)
            if val_metrics:
                val_metrics.update(
                    {
                        "epoch": trainer.current_epoch,
                        "fold": self.fold,
                        "stage": "val",
                    }
                )
                # Replace last entry if same epoch (e.g., if multiple val loops).
                if (
                    self.val_metrics
                    and self.val_metrics[-1]["epoch"] == trainer.current_epoch
                ):
                    self.val_metrics[-1] = val_metrics
                else:
                    self.val_metrics.append(val_metrics)

    def on_train_end(self, trainer, pl_module):
        if not self.val_metrics:
            return

        # Find ModelCheckpoint callback.
        ckpt_cb = next(
            (
                cb
                for cb in trainer.callbacks
                if hasattr(cb, "monitor") and hasattr(cb, "mode")
            ),
            None,
        )

        # Determine best metric.
        if ckpt_cb:
            monitor_key = ckpt_cb.monitor
            mode = ckpt_cb.mode
            if mode == "min":
                best_val = min(
                    self.val_metrics, key=lambda x: x.get(monitor_key, float("inf"))
                )
            else:
                best_val = max(
                    self.val_metrics, key=lambda x: x.get(monitor_key, float("-inf"))
                )
        else:
            rank_zero_info(
                f"[Fold {self.fold}] No ModelCheckpoint found, using last val metrics."
            )
            best_val = self.val_metrics[-1]

        # Round numeric values again just in case
        for k, v in best_val.items():
            if isinstance(v, (int, float)):
                best_val[k] = round(v, 4)

        # Only global rank 0 writes CSV.
        if self.save_csv and trainer.is_global_zero:
            df = pd.DataFrame(list(best_val.items()), columns=["metric", "value"])
            write_header = not os.path.exists(self.val_csv_path)
            df.to_csv(self.val_csv_path, mode="w", index=False, header=write_header)
            rank_zero_info(
                f"[Fold {self.fold}] Best validation metrics saved to {self.val_csv_path}"
            )


class TestMetricsLoggerCallback(Callback):
    def __init__(self, fold: int = 0):
        """Saves test metrics to a CSV file after testing completes.

        Args:
            fold (int): Current fold number (used in the filename).
        """
        super().__init__()
        self.fold = fold

    def setup(self, trainer, pl_module, stage=None):
        # Use the logger directory (e.g., exp_dir/lightning_logs/version_x).
        base_dir = trainer.logger.log_dir
        test_suffix = pl_module.test_suffix
        self.test_csv_path = os.path.join(
            base_dir, f"test_{test_suffix}_metrics_fold_{self.fold}.csv"
        )

    def on_test_end(self, trainer, pl_module):
        """
        Called when test phase ends. Saves test metrics to a CSV file.
        """
        metrics = trainer.logged_metrics

        if not metrics:
            print("Warning: No test metrics found to save.")
        else:
            # Convert tensors to scalars and round numeric values.
            clean_metrics = {}
            for k, v in metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                if isinstance(val, (int, float)):
                    val = round(val, 4)
                clean_metrics[k] = val

            # Each row is one metric.
            df = pd.DataFrame(list(clean_metrics.items()), columns=["metric", "value"])
            df.to_csv(self.test_csv_path, index=False)
            print(f"Saved test metrics to: {self.test_csv_path}")


class EpochTimerCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.perf_counter() - self.start_time
        pl_module.log(
            "epoch_duration_sec",
            duration,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # print(f"Epoch {trainer.current_epoch} duration: {duration:.2f} sec")


class BenchmarkEpochTimer(Callback):
    """Logs per-epoch duration (seconds) and saves all epoch times to a file.
    Uses CUDA events for GPU timing if available.
    """

    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_train_start(self, trainer, pl_module):
        # Prepare CUDA events if GPU available
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.cuda_available:
            self.starter.record()
        else:
            self.start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.cuda_available:
            self.ender.record()
            torch.cuda.synchronize()
            ms = self.starter.elapsed_time(self.ender)  # milliseconds
            duration = ms / 1000.0  # convert to seconds
        else:
            duration = time.perf_counter() - self.start_time

        # Save internally for later aggregation
        self.epoch_times.append(duration)

        # Log to Lightning
        pl_module.log(
            "epoch_duration_sec",
            duration,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_train_end(self, trainer, pl_module):
        # Save all epoch times to a file
        log_dir = trainer.logger.log_dir
        output_file = f"{log_dir}/epoch_times.txt"

        with open(output_file, "w") as f:
            for t in self.epoch_times:
                f.write(f"{t:.6f}\n")

        print(f"[EpochTimerCallback] Saved per-epoch durations to {output_file}")


def flatten_list(lsts: List[list]):
    """Flattents and unpacks list of lists to list of all the elements.

    Args:
        lsts: List of lists.

    Returns:
        Flattened list.
    """
    result = []
    for lst in lsts:
        result += lst
    return result


def plot_data_overview(
    dataset_groups: List[List[Any]],
    dataset_names: List[str],
    levels: Optional[List[Optional[Dict[str, List[Any]]]]] = None,
    cmaps: Optional[List[str]] = None,
    examples_per_cell: Optional[int] = 1,
    figsize_per_dataset: Tuple[float, float] = (3, 3),
    suptitle: Optional[str] = None,
    hspace: float = 0.1,
    wspace: float = 0.6,
):
    """Visualize 2-task label distributions with optional example images.

    Args:
        dataset_groups: List of dataset groups, each containing dataset versions.
        dataset_names: Names of each dataset group.
        levels: Optional list of dicts per dataset:
            {
                "y1": [0,1],             # numeric labels for matching
                "y2": [0,1],
                "y1_names": ["0-4","5-9"],  # axis labels
                "y2_names": ["thin","thick"]
            }
        cmaps: Optional list of colormaps per dataset.
        examples_per_cell: Number of example images per contingency cell.
        figsize_per_dataset: Width/height per dataset row/column in inches.
        suptitle: Optional overall figure title.
        hspace: Vertical spacing between rows.
        wspace: Horizontal spacing between columns.

    Returns:
        Matplotlib Figure object.
    """

    n_datasets = len(dataset_groups)
    n_versions = max(len(group) for group in dataset_groups)
    show_examples = bool(examples_per_cell and examples_per_cell > 0)
    n_cols = n_versions + (1 if show_examples else 0)

    fig = plt.figure(
        figsize=(figsize_per_dataset[0] * n_cols, figsize_per_dataset[1] * n_datasets)
    )

    outer = fig.add_gridspec(
        n_datasets,
        n_cols,
        hspace=hspace,
        wspace=wspace,
        width_ratios=[1.0] * n_versions + ([1.2] if show_examples else []),
    )

    levels = levels or [None] * n_datasets
    cmaps = cmaps or [None] * n_datasets

    heatmap_axes_all = {}
    top_row_axes = {}

    for i, (group, name, lvl, cmap) in enumerate(
        zip(dataset_groups, dataset_names, levels, cmaps)
    ):
        example_idx_map = None

        for j, ds in enumerate(group):
            labels = np.array([list(l) for l in ds._labels])
            table, exmap = compute_contingency_and_examples_from_labels(
                labels=labels, want_examples_per_cell=examples_per_cell or 0, levels=lvl
            )
            if example_idx_map is None and show_examples:
                example_idx_map = exmap

            row_names = (
                lvl.get("y1_names", ["Task1=0", "Task1=1"])
                if lvl
                else ["Task1=0", "Task1=1"]
            )
            col_names = (
                lvl.get("y2_names", ["Task2=0", "Task2=1"])
                if lvl
                else ["Task2=0", "Task2=1"]
            )

            col_idx = j + (1 if show_examples else 0)
            ax = fig.add_subplot(outer[i, col_idx])
            if i == 0:
                top_row_axes[col_idx] = ax

            show_y_labels = False
            if show_examples and j == 0:
                # y-axis labels on first column (example grid)
                show_y_labels = False
            elif not show_examples and j == 0:
                # y-axis labels on first heatmap
                show_y_labels = True

            plot_contingency(
                ax,
                table,
                row_names=row_names if show_y_labels else [""] * len(row_names),
                col_names=col_names,
                title=None,
            )

            if show_y_labels:
                ax.set_yticklabels(ax.get_yticklabels(), rotation=90, ha="right")
                ax.yaxis.set_tick_params(length=0)
            else:
                ax.set_yticks([])

            # Remove x-axis tick marks
            ax.tick_params(axis="x", which="both", length=0)
            heatmap_axes_all[(i, col_idx)] = ax

        # Plot examples in first column
        if show_examples:
            ax_examples = fig.add_subplot(outer[i, 0])
            plot_examples_grid(
                ax_examples, group[0], example_idx_map, cmap=cmap, title=None
            )
            ax_examples.set_xticks([])
            ax_examples.set_yticks([])
            if i == 0:
                top_row_axes[0] = ax_examples

            # Add horizontal y-axis labels along the example grid
            n_rows = len(row_names)
            y_positions = np.linspace(0.75, 0.25, n_rows)
            for y_pos, label in zip(y_positions, row_names):
                ax_examples.text(
                    -0.05,
                    y_pos,
                    label,
                    ha="right",
                    va="center",
                    rotation=90,
                    transform=ax_examples.transAxes,
                )

            # Add x-axis labels along example grid
            for x_pos, label in zip([0.25, 0.75], col_names):
                ax_examples.text(
                    x_pos,
                    -0.05,
                    label,
                    ha="center",
                    va="top",
                    rotation=0,
                    transform=ax_examples.transAxes,
                )

    # Align heatmaps to square
    if heatmap_axes_all:
        width = min(ax.get_position().width for ax in heatmap_axes_all.values())
        height = min(ax.get_position().height for ax in heatmap_axes_all.values())
        size = min(width, height)
        for ax in heatmap_axes_all.values():
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + (pos.height - size) / 2, size, size])

    # Bold column letters a,b,c aligned by column index
    arial_bold = FontProperties(weight="bold")
    for col_idx in range(n_cols):
        ax = top_row_axes[col_idx]
        bbox = ax.get_position()
        fig.text(
            bbox.x0,
            0.9,  # adjust vertical placement
            string.ascii_lowercase[col_idx],
            ha="left",
            va="bottom",
            fontproperties=arial_bold,
            fontsize=12,
        )

    if suptitle:
        fig.suptitle(suptitle, y=0.995)

    return fig


def compute_contingency_and_examples_from_labels(
    labels: np.ndarray,
    want_examples_per_cell: int = 1,
    levels: Optional[Dict[str, List[Any]]] = None,
    sample_selector: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]]]:
    """
    Compute a contingency table and example indices from labels.
    Rows = y1, Columns = y2.

    Args:
        labels: np.ndarray of shape (N, 2) where column 0 = y1, column 1 = y2.
        want_examples_per_cell: Number of examples per cell.
        levels: Optional dict {"y1": [...], "y2": [...]} for explicit ordering.
        sample_selector: Function(indices, n) → selected indices.

    Returns:
        contingency: 2D numpy array
        example_idx_map: mapping (row, col) → list of sample indices
    """
    # y1 → rows, y2 → columns
    y1 = labels[:, 0]
    y2 = labels[:, 1]

    # Convert to pandas Categorical if explicit level order is given
    if levels is not None:
        y1 = pd.Categorical(y1, categories=levels.get("y1"), ordered=True)
        y2 = pd.Categorical(y2, categories=levels.get("y2"), ordered=True)

    # Build contingency table
    contingency_df = pd.crosstab(y1, y2)

    # Ensure missing rows/columns exist
    if levels is not None:
        contingency_df = contingency_df.reindex(
            index=levels.get("y1", contingency_df.index),
            columns=levels.get("y2", contingency_df.columns),
            fill_value=0,
        )

    contingency = contingency_df.to_numpy()

    # Build example map
    example_idx_map = {}
    for r, class_y1 in enumerate(contingency_df.index):
        for c, class_y2 in enumerate(contingency_df.columns):

            # Find matching samples
            matching = np.where(
                (labels[:, 0] == class_y1) & (labels[:, 1] == class_y2)
            )[0]

            # Select examples
            if want_examples_per_cell > 0 and len(matching) > 0:
                if sample_selector is not None:
                    selected = sample_selector(matching, want_examples_per_cell)
                else:
                    selected = matching[:want_examples_per_cell]
                example_idx_map[(r, c)] = selected.tolist()
            else:
                example_idx_map[(r, c)] = []

    return contingency, example_idx_map


def plot_contingency(
    ax,
    table: np.ndarray,
    row_names=("Task1=0", "Task1=1"),
    col_names=("Task2=0", "Task2=1"),
    title="",
):
    """Plot a labeled contingency heatmap on the given axis."""
    annot = [[f"{table[r,c]}" for c in range(2)] for r in range(2)]

    sns.heatmap(
        table,
        annot=annot,
        fmt="",
        cbar=False,
        square=False,
        xticklabels=col_names,
        yticklabels=row_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cmap="Grays",
        vmin=0,
        vmax=max(1, table.sum() * 0.8),
    )

    ax.set_aspect("equal", adjustable="box")

    if title:
        ax.set_title(title, pad=8)


def plot_examples_grid(ax, dataset, example_idx_map, cmap, title=""):
    """
    Plot a 2x2 grid of example images from contingency cells,
    filling the axes completely.
    """
    ax.set_axis_off()
    if title:
        ax.set_title(title, pad=8)

    n_rows, n_cols = 2, 2

    # Each cell occupies equal fraction of the axes
    cell_w = 1.0 / n_cols
    cell_h = 1.0 / n_rows

    for r in range(n_rows):
        for c in range(n_cols):
            # inset position: (x0, y0, width, height) in axes fraction
            x0 = c * cell_w
            y0 = 1 - (r + 1) * cell_h  # top-to-bottom
            inset = ax.inset_axes([x0, y0, cell_w, cell_h])
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_aspect("auto")  # fill the cell fully

            idxs = example_idx_map.get((r, c), [])
            if idxs:
                img, *_ = (
                    dataset[idxs[0]]
                    if isinstance(dataset[idxs[0]], (list, tuple))
                    else (dataset[idxs[0]],)
                )
                inset.imshow(to_numpy_image(img), cmap=cmap, aspect="auto")
            else:
                inset.text(0.5, 0.5, "No\nsamples", ha="center", va="center")

            for spine in inset.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#888")

    # Ensure the axes fills the entire GridSpec cell
    ax.set_aspect("equal", adjustable="box")


def to_numpy_image(img):
    """Convert tensor or array to normalized NumPy image."""
    if torch.is_tensor(img):
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.detach().cpu().float()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img.permute(1, 2, 0).numpy()
        elif img.ndim == 2:
            img = img.detach().cpu().float()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img.numpy()[..., None]
        else:
            raise ValueError("Unsupported tensor shape for image.")
    else:
        import numpy as np

        img = np.asarray(img)
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = img[..., None]
    return img


def setup_fonts_and_style(style_path: Optional[str] = None):
    """Register Arial (regular + bold) locally and apply a Matplotlib style file.

    Args:
        style_path: Path to a Matplotlib style file (.mplstyle or .txt)
    """

    # Ensure local Arial fonts exist (regular + bold)
    font_dir = "fonts"
    os.makedirs(font_dir, exist_ok=True)

    fonts = {
        "Arial.ttf": "https://raw.githubusercontent.com/matomo-org/travis-scripts/master/fonts/Arial.ttf",
        "Arial-Bold.ttf": "https://raw.githubusercontent.com/matomo-org/travis-scripts/master/fonts/Arial_Bold.ttf",
    }

    for fname, url in fonts.items():
        font_path = os.path.join(font_dir, fname)

        if not os.path.isfile(font_path):
            print(f"[INFO] Downloading {fname}...")
            r = requests.get(url)
            r.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(r.content)

        # Register font with matplotlib
        font_manager.fontManager.addfont(font_path)

    # Set Arial as default font family
    plt.rcParams["font.family"] = "Arial"

    # Load explicit style file
    if style_path is None or not os.path.isfile(style_path):
        print(f"Style file not found: {style_path}")
    else:
        print(f"[INFO] Loading style file: {style_path}")
        plt.style.use(style_path)


# Function to coerce sizes to numeric
def numeric_size(size):
    if isinstance(size, (int, float)):
        return size
    return plt.rcParams["font.size"]


def scale_rcparams(
    fig_width_inch, fig_height_inch, base_width=4, base_height=4, scale_factor=None
):
    """
    Scale a figure’s text elements (ticks, labels, titles, legend) based on figure size.

    Parameters:
        fig_width_inch: figure width in inches
        fig_height_inch: figure height in inches
        base_width, base_height: reference size for scaling
        scale_factor: optional explicit scaling factor
            - if None, computed from figure size
    """
    if scale_factor is None:
        scale_factor = (fig_width_inch / base_width + fig_height_inch / base_height) / 2

    # Scale rcParams globally (optional, mostly for new figures)
    for param in plt.rcParams:
        if "size" in param or "width" in param:
            try:
                plt.rcParams[param] = plt.rcParams[param] * scale_factor
            except Exception:
                pass

    # Iterate over all existing figures
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for ax in fig.axes:
            # Tick labels
            for tick in ax.xaxis.get_major_ticks():
                label = getattr(tick, "label1", None)
                if label is not None:
                    label.set_fontsize(
                        numeric_size(label.get_fontsize()) * scale_factor
                    )
            for tick in ax.yaxis.get_major_ticks():
                label = getattr(tick, "label1", None)
                if label is not None:
                    label.set_fontsize(
                        numeric_size(label.get_fontsize()) * scale_factor
                    )

            # Axis labels (x and y)
            for label in [ax.xaxis.label, ax.yaxis.label]:
                # Use current fontsize if set, otherwise fallback to rcParams
                fs = label.get_fontsize()
                if fs is None:  # fallback if not set
                    fs = plt.rcParams.get("axes.labelsize", 12)
                label.set_fontsize(fs * scale_factor)

            # Title (slightly larger than axis labels)
            if ax.title.get_text():
                ax.title.set_fontsize(
                    numeric_size(ax.title.get_fontsize()) * scale_factor
                )
