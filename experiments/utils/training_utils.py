import os
import yaml
import ray
import datetime
import tempfile
import datetime
import tempfile
import pandas as pd

from ray.tune.logger import pretty_print, UnifiedLogger, UnifiedLogger
from ray.rllib.agents.ppo import PPOTrainer

from highway_environment.envs import Environment

repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")
configs_path = os.path.join(repo_path, "configs")


def initialize_config(env_config_path: str, model_config_path: str, train_config_path: str) -> tuple:
    from experiments.models.custom_torch_model import CustomTorchModel

    from highway_environment.envs import Environment

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

        # fixed path to stored all verification results
        base_validation_path = repo_path + "/results/validation_checkpoints/"
        
        # make sure that single verification scenario folder is used or all scenarios folders are used (take a look at self_healing.yaml)
        if train_config["validation_type"] != "mixture":
            verifications_list = [(os.path.join(base_validation_path + train_config["validation_folder_name"], train_config["validation_file_name"]), 1.0)]
        
        else:
            verifications_list = []

            # NOTE: please add new verified scenario and probabiliry percentange tuples with if-else statements if required
            if train_config["scenario_mixer"]["validation_folder_1"] is not None:
                verifications_list.append(
                    (os.path.join(base_validation_path + train_config["scenario_mixer"]["validation_folder_1"], train_config["validation_file_name"]),
                        float(train_config["scenario_mixer"]["percent_probability_1"])
                    )
                )
            if train_config["scenario_mixer"]["validation_folder_2"] is not None:
                verifications_list.append(
                    (os.path.join(base_validation_path + train_config["scenario_mixer"]["validation_folder_2"], train_config["validation_file_name"]),
                        float(train_config["scenario_mixer"]["percent_probability_2"])
                    )
                )
            if train_config["scenario_mixer"]["validation_folder_3"] is not None:
                verifications_list.append(
                    (os.path.join(base_validation_path + train_config["scenario_mixer"]["validation_folder_3"], train_config["validation_file_name"]),
                        float(train_config["scenario_mixer"]["percent_probability_3"])
                    )
                )
            if len(verifications_list) == 0:
                raise NotImplementedError("[ERROR]-> make sure that 'validation_type!=mixture' or add 'more scenarios to the mixing' if-else above!")
        
        env_configs["config"]["scenario_config"]["file_name"] = verifications_list
    
    # add environment configurations to training config
    general_config = model_configs.copy()
    general_config["env_config"] = env_configs

    # initialize environment
    env = Environment(
        config=env_configs["config"]
    )

    print("\n[INFO]-> Environment:\t", env)
    print("\n[CONFIG]-> General Configurations:\t", pretty_print(general_config))
    print("\n[CONFIG]-> Training Configurations:\t", pretty_print(train_config))
    return env, general_config, train_config


def ppo_model_initialize(general_config: dict) -> PPOTrainer:
    ray.init()

    ppo_trainer = PPOTrainer(
        config=general_config,
        env=general_config["env"],
        logger_creator=custom_log_creator(os.path.expanduser(repo_path + "/results/trained_models/"), "PPOTrainer_" + str(general_config["env"]))
    )
    
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


def custom_log_creator(custom_path: str, custom_str: str) -> UnifiedLogger:
    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)

        return UnifiedLogger(config, logdir, loggers=None)
    
    return logger_creator


def get_latest_folder_path(given_directory: str) -> str:
    all_dirs = [os.path.join(given_directory,d) for d in os.listdir(given_directory)]
    
    if len(all_dirs) == 0:
        return None
    else:
        return max(all_dirs, key=os.path.getmtime)
