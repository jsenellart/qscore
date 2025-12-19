# TNO-Quantum / qscore

## Q-score evaluation

This repository contains python code to run the Q-score (Max-Cut and Max-Clique) benchmark on the following solver backends:

- D-Wave QPU device, the `Advantage_system4.1` ~~and `DW_2000Q_6`~~ QPU system. (The `DW_2000Q_6` device no longer available.)
- D-Wave classical solvers, its `Simulated Annealing` and ~~`qbsolv`~~ `Tabu` solver. (The `qbsolv` package is deprecated, but remains available in previous version.)
- D-Wave hybrid solver.
- Gate-based hardware using QAOA on QuantumInspire and IBM hardware or simulators.
- Gaussian Boson Sampling, a form of photonic quantum computing, both simulated an using the 12-mode Quandela QPU.   
- Photonic CVaR-VQE QUBO solver derived from Quandela's in-house implementation.
- ObliQ photonic solvers (static, VQC, and hybrid circuits as in [ObliQ: Solving Quadratic Unconstrained Binary Optimization Problems on Real Photonic Quantum Machines](https://dl.acm.org/doi/10.1145/3771573).

For an introduction to the Q-score, see the reference below.
Q-score instances for Max-Cut or Max-Clique optimization problem can be run for different sizes and timeout limits. If no result is found within the allowed time limit, no objective result and a beta value of `0` is returned. Note that for the QPU solvers, the time limit considers embedding time only. The actual computation time will be slightly higher, but this difference will be in the order of milliseconds and will hence not influence the results. For similar reasons, for the photonic simulator, we only apply the time constraint to the classical runtime of the algorithm. To compute the Q-score, one runs for increasing graph size sufficiently many instances of the given code to check whether the average beta is larger than `0.2`.

This code was used to obtain results for the following papers:

- ["Evaluating the Q-score of quantum annealers", Ward van der Schoot et al. (IEEE QSW 2022)](https://ieeexplore.ieee.org/document/9860191).
- ["Q-score Max-Clique: The first metric evaluation on multiple computational paradigms", Ward van der Schoot et al. (2023 arXiv)](https://arxiv.org/abs/2302.00639)

## Q-score introduction

In December 2020, Atos introduced a new quantum metric, called the Q-score, which is supposed to be a universal quantum metric applicable to all programmable quantum processors. The idea behind Q-score is to determine how well a quantum system can solve the Max-Cut problem, which is a real-life combinatorial problem. The Q-score is determined by the maximum number of variables within such a problem that the quantum system can optimize for. For a more elaborate description, see [this paper](https://arxiv.org/abs/2102.12973). 

The official announcement of Atos can be found [here](https://atos.net/en/2020/press-release_2020_12_04/atos-announces-q-score-the-only-universal-metrics-to-assess-quantum-performance-and-superiority) and the project with tools to calculate Q-score benchmarks on gate-based devices can be found in this [gitlab project](https://github.com/myQLM/qscore).

## Usage
A single Q-score instance can be run as follows:

```python
# Run a Max-Cut problem instance of size 10 on the Advantage QPU solver of D-Wave with a time limit of 60 seconds, returning 100 reads.
python evaluate.py -p "max-cut" -s 10 -t 60 -n 100 -solver "Advantage_system4.1"
```

Multiple Q-score instances for various sizes can be run as follows:

1. Modify the following parameters accordingly in `calculate_qscore.py`
    ```python
    # Input arguments
    _NB_INSTANCES_PER_SIZE = 10
    _SIZE_RANGE = list(range(2, 9, 2))
    FILE_NAME = "example.json"
    INCLUDE_EXACT_RESULTS = True
    PROBLEM_TYPE = "max-clique"
    TIMEOUT = 60
    SOLVER = "Simulated_Annealing"
    _SEED = 49430557
    NUM_READS = 1024
    PROVIDER = None
    BACKEND = None
    _PARALLEL_WORKERS = 4
    _MIN_TIMEOUT_SIZE = 12
    _SOLVER_OPTIONS = {
        "nb_samples": 4096,
        "nb_inputs": 2,
        "platform": "sim:ascella",
        "goal": "min",
        "max_iter": 10,
        "cvar_alpha": 0.8,
    }
    ```

The `_PARALLEL_WORKERS` setting controls how many instances are executed concurrently
when launching `calculate_qscore.py`. Increase it to shorten wall-clock time at the
expense of higher CPU and memory usage; set it to `1` to run strictly sequentially.

Use `_MIN_TIMEOUT_SIZE` (or the `--min_timeout_size` flag in `evaluate.py`) to delay
process-based timeout enforcement until the problems reach a given size. Smaller
instances run inline in the main process and only check the timeout after the run
finishes, avoiding the ~3–4s overhead.

`_SOLVER_OPTIONS` (or `--solver_options '{"...": ...}'` for single runs) forwards
keyword arguments to solver-specific integrations. For `Photonic_CVARVQE` the supported
keys mirror the Quandela implementation: `nb_samples`, `nb_inputs`, `run_on_qpu`,
`run_on_gpu`, `platform`, `offset`, `goal`, `max_iter`, and `cvar_alpha`.
For ObliQ solvers, wrap coefficient-training parameters inside a `"train"` dict;
set `"optimizer": "cobyla"` to switch from the default Adam loop to gradient-free COBYLA updates.

2. Run the `calculate_qscore` script. A json file with results will be created inside the `data` folder.
    ```python
    python calculate_qscore.py
    ```

3. The beta vs N and time vs N Q-score graphs can be plotted for a given results file by running `plot_qscore.py`:

    ```python
    # Plot Q-score graph for results file
    python plot_qscore.py -f "example.json" -e
    ```

    Plot overlays can be customized via the following optional flags:

    - `--show_minmax` fills the min/max envelope for beta and time.
    - `--minmax_lines` draws dotted min/max borders (with or without the fill).
    - `--show_stddev` shades the +/-1σ band around the mean curves.
    - `--stddev_lines` adds dotted +/-1σ boundaries; useful when shading is disabled.
    - `--log_time` switches the elapsed-time axis to a logarithmic scale.

    Combine the toggles to keep multi-curve figures readable (for example,
    `--minmax_lines --stddev_lines` to show only dotted bounds, or
    `--show_minmax --show_stddev` for filled bands).

### Timeout behavior

The `-t/--timeout` flag in `evaluate.py` and `calculate_qscore.py` enforces a
real execution timeout for the corresponding solvers by running each instance in its own
process. When the deadline is hit the child process is terminated and the
instance is counted as `beta = 0`. Spawning these short-lived worker processes
introduces a constant overhead (around 3–4 seconds); take this into account when choosing tight timeout values or when running large batches.
You can defer this process-based enforcement to larger problem sizes by setting
`_MIN_TIMEOUT_SIZE` (batch runs) or `--min_timeout_size` (single runs); smaller problems
will execute inline and only enforce the timeout after completion.

### Photonic CVaR-VQE solver

The `Photonic_CVARVQE` solver wraps Quandela's optimized CVaR-VQE routine for QUBO
instances. Supply its parameters via `_SOLVER_OPTIONS`/`--solver_options`, for example:

```python
python evaluate.py -p "max-cut" -s 12 -t 60 -solver "Photonic_CVARVQE" \
    --solver_options '{"nb_samples": 4096, "nb_inputs": 2, "platform": "sim:ascella"}'
```

Only Max-Cut and Max-Clique workloads are supported; the solver converts the sampled
bit strings back into Q-score objectives (cut size or clique size) before reporting
`beta` values.

### ObliQ photonic solvers

The ObliQ static, VQC, and hybrid solvers reproduce the circuits described in
["ObliQ: Solving Quadratic Unconstrained Binary Optimization Problems on Real Photonic Quantum Machines"](https://dl.acm.org/doi/10.1145/3771573) (SIGMETRICS 2025, Aditya Ranjan *et al.*).
Select them via `-solver obliq-static`, `obliq-vqc`, or `obliq-hybrid`. All three share
the same solver options:

- `nsamples`, `num_rep`, `graph_mode`: tune sampling depth, anchor repetitions, and
    the heuristic used in `solution_guesses`. Defaults are `nsamples = 5000`,
    `num_rep = 10`, `graph_mode = 0`.
- `backend`, `token`, `real_machine`: forward choices to Perceval (use `real_machine`
    or set `backend` like `qpu:ascella`). Defaults are `backend = None`, `token = None`,
    `real_machine = False`.
- `coeffs`: initial VQC parameter vector (optional; defaults to zeros when omitted).
- `train`: dictionary enabling coefficient optimization before each evaluation
    (disabled unless present). Defaults for the trainer are `optimizer = "adam"`,
    `max_iter = 5`, `learning_rate = 0.05`, `finite_diff_step = pi/4`,
    `beta1 = 0.9`, `beta2 = 0.999`, and `epsilon = 1e-8`. Set
    `{"optimizer": "cobyla"}` for a gradient-free run.

Example CLI usage:

```bash
python evaluate.py -p max-cut -s 8 -solver obliq-hybrid --solver_options '{
        "nsamples": 4096,
        "num_rep": 12,
        "train": {"max_iter": 5, "optimizer": "cobyla"}
}'
```

## Configuration

To use this code we assume that the reader has installed the requirements and set up access to the required solvers. 

Requirements can be installed using pip and have been tested for `python3.9`, `python3.10` and `python3.13`:
```terminal
python -m pip install -r requirements.txt
```

### Set up D-Wave configuration
To use the quantum annealing or hybrid solvers we assume the reader has created a D-Wave Leap account and configured access to D-Wave's solvers correctly. To make an account, please visit the [website of D-wave](https://cloud.dwavesys.com/leap/login/?next=/leap/). To configure access to the solvers, please visit [these instructions](https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html).

Example usage for the `Advantage_system4.1` QPU solver:

```python
python evaluate.py -p "max-cut" -s 10 -t 60 -n 100 -solver "Advantage_system4.1"
```
### Set up QuantumInspire configuration

To use the Quantum Inspire backend we assume the reader has created a QuantumInspire account and configured access correctly.

1. Create a Quantum Inspire account (https://www.quantum-inspire.com/)
2. Get an API token from the Quantum Inspire website.
3. With your API token run: 

```python
from quantuminspire.credentials import save_account
save_account('YOUR_API_TOKEN')
```

Example usage for the `Starmon-5` hardware backend:
```python
python evaluate.py -p "max-clique" -s 5 -t 60 -n 1024 -solver "QAOA" -provider "qi" -backend "Starmon-5" 
```

### Set up IBM configuration

To use IBM hardware backends, one need to create and IBM Quantum account, which can be done [here](https://quantum-computing.ibm.com/lab). 
After creating your account you can install your API key using the Qiskit Runtime service:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account('MY_API_TOKEN')
```

Example usage for the IBM Lima device:
```python
python evaluate.py -p "max-clique" -s 5 -t 60 -n 1024 -solver "QAOA" -provider "ibm" -backend "ibmq_lima" 
```

### Set up Quandela configuration

To use the Quandela hardware backend, one need to create and Quandela Cloud account, which can be done [here](https://cloud.quandela.com/webide/login). 
After creating your account you can store your API key inside the `configuration/_QUANDELA_API_KEY`:

Example usage for the photonic Quandela Ascella hardware:
```python
python evaluate.py -p "max-clique" -s 3 -t 60 -n 1024 -solver "Photonic_quandela" -backend "qpu:ascella" 
```