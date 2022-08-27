# Black-Box-Verification

Test Verification Algorithms on Gym Highway Environment using Ray RLlib

* [Installation](#installation)
    - [Export Repository Path](#export-repository-path)
    - [Anaconda Environment Creation](#anaconda-environment-creation)
    - [Environment Installation](#environment-installation)
    - [Package Installation](#package-installation)
* [Tests](#tests)
    - [Test Training Example](#test-training-example)
    - [Train RL Agent](#train-rl-agent)
* [Evaluation](#evaluation)
    - [Evaluate RL Agent](#evaluate-rl-agent)
* [Verification Algorithms](#verification-algorithms)
    - [Grid-Search Validation](#grid-search-validation)
    - [Monte-Carlo-Search Validation](#monte-carlo-search-validation)
    - [Cross-Entropy-Search Validation](#cross-entropy-search-validation)
    - [Bayesian-Optimization-Search Validation](#bayesian-optimization-search-validation)
    - [Adaptive-Sequential-Monto-Carlo-Search Validation](#adaptive-sequential-monto-Carlo-search-validation)
* [Self Healing](#self-healing)
    - [Train RL on Custom Verification Scenarios](#train-rl-on-custom-verification-scenarios)
* [Analyse Results](#analyse-results)
    - [Analyse & Visualize Validation Scenarios](#analyse-&-visualize-validation-scenarios)
* [Sbatch Slurm](#sbatch-slurm)
    - [Slurm Training & Verification](#slurm-training-&-verification)

# Installation

---
## Export Repository Path

> save this directory to .bashrc
```sh
gedit ~/.bashrc
```

> paste and save the following to .bashrc file
```sh
export BLACK_BOX="LocalPathOfThisRepository"
```

> execure saved changes
```sh
source ~/.bashrc
```

---
## Anaconda Environment Creation

> used python3.7
```python
conda create -n highway python=3.7

conda activate highway
```

> install required packages
```python
pip install -r requirements.txt
```

In order to successfully use GPUs, please install CUDA by following the site : https://pytorch.org/get-started/locally/

---
## Environment Installation

> prepare Ubuntu
```python
sudo apt-get update -y

sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev
    python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

> install highway-environment
```python
pip install highway-env==1.5
```

> register custom environment wrapper class
```sh
cd highway_environment/highway_environment

python create_env.py
```

> install custom highway environment globally
```sh
cd highway_environment

python setup.py install
```
NOTE: make sure that after each update of Environment class, environment installation has to be performed again

---
## Package Installation

> install Ray + dependencies for Ray Tune
```python
pip install -U "ray[tune]"
```

> install Ray + dependencies for Ray RLlib
```python
pip install -U "ray[rllib]"
```

# Tests

---
## Test Training Example

> run default PPO training example with ray.tune
```sh
cd highway_environment/highway_environment

python test_train.py
```

---
## Train RL Agent

NOTE:
* parameters of a trained model will be saved at **~/ray_results/** folder
* please change _train-or-eval_ parameter to _true_ inside **/experiments/configs/train_config.yaml** config

```sh
cd experiments/training

python ppo_train.py
```

# Evaluation

---
## Evaluate RL Agent

NOTE:
* parameters of a trained model should be moved to **/experiments/results/trained_models/** folder from **~/ray_results/** folder
* please check _load-agent-name_ key inside **/experiments/configs/evaluation_config.yaml** config to be the model intended to evaluate
* consider _initial-space_ key in the same config yaml that represents the initial conditions of the front vehicle while evaluation

```sh
cd experiments/evaluation

python evaluate_model.py
```

# Verification Algorithms

---
## Grid-Search Validation

> apply grid-search algorithm verification on a trained rl model

NOTE:
* check _load-agent-name_ key inside **/experiments/configs/grid_search.yaml** config and make sure that the model is located in **/experiments/results/trained_models/** folder

```sh
cd experiments/algorithms

python grid_search.py
```

---
## Monte-Carlo-Search Validation

> apply monte-carlo-search algorithm verification on a trained rl model

NOTE:
* check _load-agent-name_ key inside **/experiments/configs/monte_carlo_search.yaml** config and make sure that the model is located in **/experiments/results/trained_models/** folder

```sh
cd experiments/algorithms

python monte_carlo_search.py
```

---
## Cross-Entropy-Search Validation

> apply cross-entropy-search algorithm verification on a trained rl model

NOTE:
* check _load-agent-name_ key inside **/experiments/configs/ce_search.yaml** config and make sure that the model is located in **/experiments/results/trained_models/** folder
* check _number-of-samples_ key inside **/experiments/configs/ce_search.yaml** config is defined as a multiplication of iteration number and sample size per iteration. At each iteration best 10 percent will be selected from batch of sample size to determine next iteration's minimum and maximum limits.

```sh
cd experiments/algorithms

python ce_search.py
```

---
## Bayesian-Optimization-Search Validation

> apply bayesian-optimization-search algorithm verification on a trained rl model

> install package
```sh
pip install bayesian-optimization
```

NOTE:
* check _load-agent-name_ key inside **/experiments/configs/bayesian_search.yaml** config and make sure that the model is located in **/experiments/results/trained_models/** folder

```sh
cd experiments/algorithms

python bayesian_search.py
```

---
## Adaptive-Sequential-Monto-Carlo-Search Validation

> apply adaptive-sequential-monte-carlo-search algorithm verification on a trained rl model

NOTE:
* check _load-agent-name_ key inside **/experiments/configs/adaptive_seq_mc_search.yaml** config and make sure that the model is located in **/experiments/results/trained_models/** folder

```sh
cd experiments/algorithms

python adaptive_seq_mc_search.py
```

# Self-Healing

---
## Train RL on Custom Verification Scenarios

### Simple Verification
> after applying verification algorithm, RL agent could be trained again on validation results

NOTE:
- use **/experiments/configs/self_healing.yaml** config
- train model with **/experiments/training/self_healing.py** script
- trained model could be loaded and re-trained from latest checkpoint with _is-restore_ key inside config
- custom scenario setter class is located at **/experiments/utils/scenarios.py**
- new scenario loader could be added and referenced with key _validation-type_ in config **self_healing.yaml** config

### Probabilistic Verification
> to include specific verification results into sampling container, read the following note

NOTE:
- change _validation-type_ key inside **/experiments/configs/self_healing.yaml** config to "mixture"
- take a look at _scenario-mixer_ key parameters and specify which validation results to include
- each validation comes with probability of sampling which should sum up to **1.0**
- folder names in _scenario-mixer_ key should be **null** if not specified and total sum _percentage-probability_ of existing folders should be 100 (1.0)

```sh
cd experiments/training

python self_healing.py
```

# Analyse Results

---
## Analyse & Visualize Validation Scenarios

> after training and running a verification algorithm, visualize validation and failure scenarios

```sh
cd experiments/analyses

python3 -m notebook
```

# Sbatch Slurm

---
## Slurm Training & Verification

> submit a batch script to slurm for training an RL model

```sh
cd experiments/training

conda activate highway

# checkout resource allocations before submitting a slurm batch
sbatch slurm_train.sh
```

> submit a batch script to slurm for applying selected verification algorithm

```sh
cd experiments/algorithms

conda activate highway

# checkout selected algorithm and resource allocations before submitting a slurm batch
sbatch slurm_verification.sh
```

> basic slurm commands

```sh
# submit a batch script to Slurm for processing
sbatch <job-id>

# show information about your job(s) in the queue
squeue

# show information about current and previous jobs
sacct

# end or cancel a queued job
scancel <job-id>

# read last lines of terminal logs (.err or .out)
tail -f <job-id>.out
```