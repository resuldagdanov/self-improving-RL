# Black-Box-Verification

Test Verification Algorithms on Gym Highway Environment using Ray RLlib

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
```python
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

---
## Evaluate RL Agent

NOTE:
* parameters of a trained model should be moved to **/experiments/results/trained_models/** folder from **~/ray_results/** folder
* please change _train-or-eval_ parameter to false_ inside **/experiments/configs/train_config.yaml** config

```sh
cd experiments/training

python ppo_train.py
```
