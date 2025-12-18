"""Plot beta vs N and time vs N Q-score graphs for one or more datasets."""
import argparse
import json
import os
from itertools import cycle
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.approximation.maxcut import one_exchange

from utils.max_clique import calculate_beta_max_clique, naive_clique_size
from utils.max_cut import calculate_beta_max_cut


def parse_args() -> argparse.Namespace:
    """
    Parser function.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        nargs="+",
        help="One or more data files (within /data unless absolute path)",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--exact",
        action="store_true",
        help="Use graph instances to calculate beta",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--time_constraint",
        action="store_true",
        help="Ignore time constraint",
        required=False,
    )
    parser.add_argument(
        "--show_envelope",
        action="store_true",
        help="Display min/max envelope shading",
        required=False,
    )
    args = parser.parse_args()
    return args


def _resolve_file_path(file_arg: str) -> str:
    """Resolve user-provided file path to an existing location."""

    if os.path.exists(file_arg):
        return file_arg
    candidate = os.path.join("data", file_arg)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Unable to locate data file: {file_arg}")


def _prepare_dataset(
    data: dict, exact: bool, time_constraint: bool
) -> Optional[dict]:
    """Compute beta/time statistics for a dataset."""

    problem_range = np.array(
        sorted(
            int(k)
            for k, v in data.items()
            if k != "settings"
            and isinstance(v, dict)
            and v.get("result") not in (None, [])
        )
    )
    if problem_range.size == 0:
        return None

    problem_type = data["settings"]["PROBLEM_TYPE"]
    solver = data["settings"]["SOLVER"]

    mins_beta, maxes_beta, means_beta, stds_beta = [], [], [], []
    mins_time, maxes_time, means_time, stds_time = [], [], [], []
    plotted_sizes = []

    if exact:
        seed = data["settings"]["SEED"]
        for size in problem_range:
            print(f"Starting exact calculation for size: {size}")
            results = np.array(data[str(size)]["result"])
            times = np.array(data[str(size)]["times"])

            if len(results) == 0:
                continue

            graphs = []
            for _ in range(len(results)):
                G = nx.erdos_renyi_graph(size, 1 / 2, seed=seed)
                graphs.append(G)
                seed += 1

            if "exact-result" in data[str(size)]:
                exact_results = np.array(data[str(size)]["exact-result"])
            else:
                exact_results = np.array(
                    [
                        one_exchange(G)[0]
                        if problem_type == "max-cut"
                        else nx.max_weight_clique(G, weight=None)[1]
                        for G in graphs
                    ]
                )

            betas = []
            if problem_type == "max-cut":
                for result, exact_result in zip(results, exact_results):
                    random_score = size * (size - 1) / 8
                    if random_score == exact_result:
                        betas.append(1)
                    else:
                        betas.append((result - random_score) / (exact_result - random_score))

            elif problem_type == "max-clique":
                for result, exact_result, G in zip(results, exact_results, graphs):
                    random_score = np.average([naive_clique_size(G) for _ in range(1000)])
                    if random_score == exact_result:
                        betas.append(1)
                    else:
                        betas.append((result - random_score) / (exact_result - random_score))

            betas = np.array(betas)
            mins_beta.append(betas.min())
            maxes_beta.append(betas.max())
            means_beta.append(betas.mean())
            stds_beta.append(betas.std())

            mins_time.append(times.min())
            maxes_time.append(times.max())
            means_time.append(times.mean())
            stds_time.append(times.std())
            plotted_sizes.append(size)

    else:
        calculate_beta_function = (
            calculate_beta_max_cut
            if problem_type == "max-cut"
            else calculate_beta_max_clique
        )
        for size in problem_range:
            results = np.array(data[str(size)]["result"])
            times = np.array(data[str(size)]["times"])
            times = times[~np.isnan(results)]
            results = results[~np.isnan(results)]

            if len(results) == 0:
                continue

            betas = np.array([calculate_beta_function(size, result) for result in results])
            mins_beta.append(betas.min())
            maxes_beta.append(betas.max())
            means_beta.append(betas.mean())
            stds_beta.append(betas.std())

            mins_time.append(times.min())
            maxes_time.append(times.max())
            means_time.append(times.mean())
            stds_time.append(times.std())
            plotted_sizes.append(size)

    if not plotted_sizes:
        return None

    mins_beta, maxes_beta, means_beta, stds_beta = (
        np.array(mins_beta),
        np.array(maxes_beta),
        np.array(means_beta),
        np.array(stds_beta),
    )
    mins_time, maxes_time, means_time, stds_time = (
        np.array(mins_time),
        np.array(maxes_time),
        np.array(means_time),
        np.array(stds_time),
    )
    problem_range = np.array(plotted_sizes)

    if time_constraint:
        feasible_mask = means_beta > 0.2
    else:
        feasible_mask = (means_beta > 0.2) & (means_time < 60)

    feasible_sizes = problem_range[feasible_mask]
    qscore = int(feasible_sizes.max()) if feasible_sizes.size else None

    return {
        "problem_range": problem_range,
        "mins_beta": mins_beta,
        "maxes_beta": maxes_beta,
        "means_beta": means_beta,
        "stds_beta": stds_beta,
        "mins_time": mins_time,
        "maxes_time": maxes_time,
        "means_time": means_time,
        "stds_time": stds_time,
        "problem_type": problem_type,
        "solver": solver,
        "qscore": qscore,
    }


def plot_graphs(
    files: list[str],
    exact: Optional[bool] = False,
    time_constraint: Optional[bool] = False,
    show_envelope: bool = False,
) -> None:
    """
    Plot Q-score graphs. Both the beta vs N and time vs N graphs are plotted.

    Args:
        files: Paths to data files.
        exact: Whether to include exact results in beta calculation.
            Only suitable for small problem sizes.
    """
    datasets = []
    for file_arg in files:
        try:
            file_path = _resolve_file_path(file_arg)
        except FileNotFoundError as exc:
            print(exc)
            continue

        with open(file_path) as json_file:
            data = json.load(json_file)

        stats = _prepare_dataset(data, exact, time_constraint)
        if stats is None:
            print(f"No valid data points found in {file_arg}; skipping.")
            continue

        stats["file_path"] = file_path
        stats["label"] = f"{stats['solver']} ({os.path.basename(file_path)})"
        datasets.append(stats)

    if not datasets:
        print("No datasets available for plotting.")
        return

    problem_types = {dataset["problem_type"] for dataset in datasets}
    if len(problem_types) != 1:
        raise ValueError("All datasets must have the same PROBLEM_TYPE to be plotted together.")
    problem_type = problem_types.pop()

    all_sizes = sorted({size for dataset in datasets for size in dataset["problem_range"]})
    x_min = max(min(all_sizes) - 1, 0)
    x_max = max(all_sizes) + 1
    time_ylim = min(90, max(np.max(dataset["maxes_time"]) for dataset in datasets) * 1.2)
    if show_envelope:
        beta_max_candidate = max(np.max(dataset["maxes_beta"]) for dataset in datasets)
    else:
        beta_max_candidate = max(
            np.max(dataset["means_beta"] + dataset["stds_beta"]) for dataset in datasets
        )
    beta_ylim_upper = max(0.5, min(1.5, beta_max_candidate * 1.1))

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f"Q-score {problem_type} comparison ({len(datasets)} configurations)")

    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"]))
    for dataset in datasets:
        color = next(color_cycle)
        problem_range = dataset["problem_range"]
        qscore_label = dataset["qscore"] if dataset["qscore"] is not None else "N/A"
        beta_line_label = f"{dataset['label']} (Q={qscore_label})"

        if show_envelope:
            axs[0].fill_between(
                problem_range,
                dataset["mins_beta"],
                dataset["maxes_beta"],
                color=color,
                alpha=0.1,
            )

        beta_lower_std = dataset["means_beta"] - dataset["stds_beta"]
        beta_upper_std = dataset["means_beta"] + dataset["stds_beta"]
        if exact:
            beta_upper_std = np.minimum(beta_upper_std, 1.0)

        axs[0].fill_between(
            problem_range,
            beta_lower_std,
            beta_upper_std,
            color=color,
            alpha=0.2,
        )
        axs[0].plot(
            problem_range,
            dataset["means_beta"],
            color=color,
            marker="o",
            linewidth=2,
            label=beta_line_label,
        )

        if show_envelope:
            axs[1].fill_between(
                problem_range,
                dataset["mins_time"],
                dataset["maxes_time"],
                color=color,
                alpha=0.1,
            )
        axs[1].fill_between(
            problem_range,
            np.maximum(dataset["means_time"] - dataset["stds_time"], 0),
            dataset["means_time"] + dataset["stds_time"],
            color=color,
            alpha=0.2,
        )
        axs[1].plot(
            problem_range,
            dataset["means_time"],
            color=color,
            marker="o",
            linewidth=2,
            label=dataset["label"],
        )

    axs[0].set_title(f"Beta {'(exact)' if exact else ''}")
    axs[0].set(
        xlabel="Problem size N",
        ylabel="Beta",
        xlim=[x_min, x_max],
        ylim=[-0.3, beta_ylim_upper],
    )
    axs[0].set_xticks(all_sizes)
    axs[0].axhline(y=0.2, color="r", linestyle="--")

    axs[1].set_title("Elapsed time")
    axs[1].set(
        xlabel="Problem size N",
        ylabel="Time (in s)",
        xlim=[x_min, x_max],
        ylim=[0, max(time_ylim, 1)],
    )
    axs[1].set_xticks(all_sizes)
    axs[1].axhline(y=60, color="r", linestyle="--")

    axs[0].legend(loc="lower right")
    axs[1].legend(loc="upper left")

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    exact = args.exact
    time_constraint = args.time_constraint

    plot_graphs(args.files, exact, time_constraint, args.show_envelope)
