import os
import yaml
import ray

from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer

from highway_environment.envs import Environment

repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")
configs_path = os.path.join(repo_path, "configs")


def initialize_config(env_config_path: str, model_config_path: str, train_config_path: str) -> tuple:
    # highway environment configirations
    with open(configs_path + env_config_path) as f:
        env_configs = yaml.safe_load(f)
    
    # training algorithms configurations
    with open(configs_path + model_config_path) as f:
        model_configs = yaml.safe_load(f)
    
    # general parameters for training
    with open(configs_path + train_config_path) as f:
        train_config = yaml.safe_load(f)
    
    # add environment configurations to training config
    general_config = model_configs.copy()
    general_config["env_config"] = env_configs

    # initialize environment
    env = Environment(config=env_configs["config"])

    print("\n[INFO]-> Environment:\t", env)
    print("\n[CONFIG]-> General Configurations:\t", pretty_print(general_config))
    print("\n[CONFIG]-> Training Configurations:\t", pretty_print(train_config))
    return env, general_config, train_config


def ppo_model_initialize(general_config: dict) -> object:
    ray.init()
    ppo_trainer = PPOTrainer(config=general_config, env=general_config["env"])

    print("\n[INFO]-> PPO Trainer:\t", ppo_trainer)
    return ppo_trainer
