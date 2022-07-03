import os
import pickle
import yaml
import ray

from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print

from highway_environment.envs import Environment


def initialize_config(env_config_path: str, model_config_path: str) -> tuple:
    # highway environment configirations
    with open(env_config_path) as f:
        env_configs = yaml.safe_load(f)
    
    # training algorithms configurations
    with open(model_config_path) as f:
        model_configs = yaml.safe_load(f)
    
    # add environment configurations to training config
    general_config = model_configs.copy()
    general_config["env_config"] = env_configs

    # initialize environment
    env = Environment(config=env_configs["config"])

    return env, general_config


def initialize_model(general_config: dict) -> object:
    ray.init()
    trainer = PPOTrainer(config=general_config, env=general_config["env"])

    return trainer


if __name__ == "__main__":
    seed = 12

    # get directory paths
    repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")
    configs_path = os.path.join(repo_path, "configs")

    # organize parameters
    env, general_config = initialize_config(
        env_config_path=configs_path + "/env_config.yaml",
        model_config_path=configs_path + "/ppo_config.yaml"
    )

    # define PPO agent trainer
    trainer = initialize_model(
        general_config=general_config
    )

    print("trainer : ", trainer)
