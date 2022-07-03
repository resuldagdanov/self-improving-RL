import os
import yaml
import numpy as np

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from highway_environment.envs import Environment


class MainAgent:

    def __init__(self, algorithm_config: dict) -> None:
        self.repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")

        self.max_eps_length = algorithm_config["max_eps_length"]
        self.experiment_name = algorithm_config["experiment_name"]

    def __str__(self) -> str:
        return "Agent Folder: %s\n", "Checkpoint Number: %d"
    
    @staticmethod
    def create_search_space(params: dict) -> tuple:
        distance_space = list(
            np.linspace(
                start   =   params['distance']['min'],
                stop    =   params['distance']['max'],
                num     =   params['distance']['steps'],
                endpoint=   True,
                retstep =   True,
                dtype   =   np.float32
            )
        )
        velocity_space = list(
            np.linspace(
                start   =   params['velocity']['min'],
                stop    =   params['velocity']['max'],
                num     =   params['velocity']['steps'],
                endpoint=   True,
                retstep =   True,
                dtype   =   np.float32
            )
        )

        return distance_space, velocity_space
    
    def initialize_config(self, env_config_path: str, algo_config_path: str) -> dict:
        # highway environment configirations
        with open(env_config_path) as f:
            env_configs = yaml.safe_load(f)

        # training algorithms configurations
        with open(algo_config_path) as f:
            algo_configs = yaml.safe_load(f)
        
        # add environment configurations to training config
        general_config = model_configs.copy()
        general_config["train"]["config"]["env_config"] = env_configs

        return general_config

    def initialize_model(self, params: dict):
        pass

    def initialize_environment(self, config: dict):
        pass

    def run_episode(self, env: Environment, agent: PPOTrainer, config: dict):
        pass

    def simulate(self, params: dict) -> None:
        pass
