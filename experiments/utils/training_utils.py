import os
import yaml
import ray
import pandas as pd

from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer

from highway_environment.envs import Environment

repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")
configs_path = os.path.join(repo_path, "configs")

from experiments.models.custom_torch_model import CustomTorchModel


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
    
    # number of ray resources are set from training configuration
    model_configs["num_gpus"] = train_config["num_gpus"]
    model_configs["num_envs_per_worker"] = train_config["num_envs_per_worker"]
    model_configs["num_cpus_per_worker"] = train_config["num_cpus_per_worker"]
    model_configs["num_gpus_per_worker"] = train_config["num_gpus_per_worker"]
    
    # remove workers while render is open
    if env_configs["config"]["rendering"]:
        model_configs["num_workers"] = 0
    
    # use custom nn model if required
    if train_config["use_custom_torch_model"] is True:
        model_configs["model"]["custom_model"] = "CustomTorchModel" # TODO: change this when new model is implemented

        model_name = model_configs["model"]["custom_model"]
        if model_name == "CustomTorchModel":
            ray.rllib.models.ModelCatalog.register_custom_model(model_name, CustomTorchModel)
        else:
            print("\n[ERROR]-> Custom Model Named:\t", model_name, "is Not Supported Yet")
    
    # set custom scenario loader attributes
    if "validation_folder_name" in train_config:
        env_configs["config"]["scenario_config"]["type"] = train_config["validation_type"]
        
        env_configs["config"]["scenario_config"]["file_name"] = \
            os.path.join(repo_path + "/results/validation_checkpoints/" \
                + train_config["validation_folder_name"], train_config["validation_file_name"])
    
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


def extract_progress_csv(file_path: str) -> pd.DataFrame:
    file_df = pd.read_csv(file_path)

    # parameters should be included in progress.csv inside folders of ./trained_models
    filtered_df = file_df[[
            "num_agent_steps_trained",
            "training_iteration",
            "hist_stats/episode_reward",
            "hist_stats/episode_lengths",
            "info/learner/default_policy/learner_stats/policy_loss",
            "info/learner/default_policy/learner_stats/vf_loss",
            "info/learner/default_policy/learner_stats/kl",
            "info/learner/default_policy/learner_stats/entropy"
        ]
    ]
    return filtered_df
