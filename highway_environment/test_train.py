import yaml

from ray.tune.tune import run_experiments

# path to default configurations
env_config_path = "./default_configs/env_config.yaml"
train_config_path = "./default_configs/train_config.yaml"

# default highway environment config
with open(env_config_path) as f:
    env_configs = yaml.safe_load(f)

# default PPO training config
with open(train_config_path) as f:
    train_configs = yaml.safe_load(f)

# add environment configurations to training config
general_config = train_configs.copy()
general_config["train"]["config"]["env_config"] = env_configs

# example experiment
run_experiments(general_config, concurrent=True, resume=False)
