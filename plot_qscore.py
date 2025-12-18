"""
Plot beta vs N and time vs N Q-score graphs.
"""
import argparse
import json
import os
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
        help="Name of data file. (within /data folder)",
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
    args = parser.parse_args()
    return args


def plot_graph(
    file: str, exact: Optional[bool] = False, time_constraint: Optional[bool] = False
) -> None:
    """
    Plot Q-score graphs. Both the beta vs N and time vs N graphs are plotted.

    Args:
        file: Path to data file (within /data folder).
        exact: Whether to include exact results in beta calculation.
            Only suitable for small problem sizes.
    """
    # Load data from json file.
    with open(file) as json_file:
        data = json.load(json_file)

    problem_range = np.array(
        sorted(
            int(k)
            for k, v in data.items()
            if k != "settings" and isinstance(v, dict) and v.get("result") not in (None, [])
        )
    )
    if problem_range.size == 0:
        print("No completed problem sizes found in data; skipping plot.")
        return
    problem_type = data["settings"]["PROBLEM_TYPE"]
    solver = data["settings"]["SOLVER"]

    # Do beta-calculations
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

            graphs = []  # Calculate graphs from seed
            for _ in range(len(results)):
                G = nx.erdos_renyi_graph(size, 1 / 2, seed=seed)
                graphs.append(G)
                seed += 1

            if "exact-result" in data[str(size)]:
                exact_results = np.array(data[str(size)]["exact-result"])
            else:  # Calculate exact result from graph
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
                        beta = (result - random_score) / (exact_result - random_score)
                        betas.append(beta)

            elif problem_type == "max-clique":
                for result, exact_result, G in zip(results, exact_results, graphs):
                    random_score = np.average(
                        [naive_clique_size(G) for _ in range(1000)]
                    )
                    if random_score == exact_result:
                        betas.append(1)
                    else:
                        beta = (result - random_score) / (exact_result - random_score)
                        betas.append(beta)

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

            betas = np.array(
                [calculate_beta_function(size, result) for result in results]
            )
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
        print("No valid data points found after filtering; skipping plot.")
        return

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

    # Create plots:
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    qscore_label = qscore if qscore is not None else "N/A"
    fig.suptitle(f"Q-score {problem_type} = {qscore_label} for solver: {solver}")

    # Beta-plot
    axs[0].fill_between(
        problem_range,
        mins_beta,
        maxes_beta,
        color="tab:blue",
        alpha=0.15,
        label="min-max range",
    )

    beta_lower_std = means_beta - stds_beta
    beta_upper_std = means_beta + stds_beta
    if exact:
        beta_upper_std = np.minimum(beta_upper_std, 1.0)

    axs[0].fill_between(
        problem_range,
        beta_lower_std,
        beta_upper_std,
        color="tab:gray",
        alpha=0.3,
        label="±1σ",
    )
    axs[0].plot(
        problem_range,
        means_beta,
        color="black",
        marker="o",
        linewidth=2,
        label="mean beta",
    )
    axs[0].set_title(f"Beta {'(exact)' if exact else ''}")
    x_min = max(problem_range.min() - 1, 0)
    x_max = problem_range.max() + 1
    axs[0].set(
        xlabel="Problem size N",
        ylabel="Beta",
        xlim=[x_min, x_max],
        ylim=[-0.3, 1.5],
    )
    axs[0].set_xticks(problem_range)
    axs[0].axhline(y=0.2, color="r", linestyle="--")

    # Time-plot
    axs[1].set_title("Elapsed time")
    axs[1].set(
        xlabel="Problem size N",
        ylabel="Time (in s)",
        xlim=[x_min, x_max],
        ylim=[0, min(90, max(maxes_time) * 1.2)],
    )
    axs[1].set_xticks(problem_range)
    axs[1].fill_between(
        problem_range,
        mins_time,
        maxes_time,
        color="tab:blue",
        alpha=0.15,
        label="min-max range",
    )

    axs[1].fill_between(
        problem_range,
        np.maximum(means_time - stds_time, 0),
        means_time + stds_time,
        color="tab:gray",
        alpha=0.3,
        label="±1σ",
    )
    axs[1].plot(
        problem_range,
        means_time,
        color="black",
        marker="o",
        linewidth=2,
        label="mean time",
    )
    axs[1].axhline(y=60, color="r", linestyle="--")
    axs[0].legend(loc="lower right")
    axs[1].legend(loc="upper left")

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    file = f"data{os.sep}" + args.file
    exact = args.exact
    time_constraint = args.time_constraint

    plot_graph(file, exact, time_constraint)
