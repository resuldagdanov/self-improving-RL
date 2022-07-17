import os
import sys
import yaml
import numpy as np

from typing import Optional
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer

from highway_environment.envs import Environment

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

from experiments.utils import validation_utils


class MainAgent:

    def __init__(self, algorithm_config: dict) -> None:
        self.repo_path = validation_utils.repo_path
        self.algorithm_config = algorithm_config
    
    @staticmethod
    def create_search_space(params: dict) -> tuple:
        distance_space = list(
            np.linspace(
                start   =   params["distance"]["min"],
                stop    =   params["distance"]["max"],
                num     =   params["distance"]["steps"],
                dtype   =   np.float32
            )
        )
        velocity_space = list(
            np.linspace(
                start   =   params["velocity"]["min"],
                stop    =   params["velocity"]["max"],
                num     =   params["velocity"]["steps"],
                dtype   =   np.float32
            )
        )

        return distance_space, velocity_space
    
    def initialize_config(self, env_config_path: str, search_config: dict, model_config_path: str) -> dict:
        # highway environment configirations
        with open(validation_utils.configs_path + env_config_path) as f:
            env_configs = yaml.safe_load(f)
        
        # change vehicle initialization parameters according to search space configuration
        env_configs["config"]["set_manually"]["ego_position"] = [ 0.0, 0.0 ]
        env_configs["config"]["set_manually"]["ego_heading"] = 0.0
        env_configs["config"]["set_manually"]["ego_speed"] = search_config["ego_v1"]
        env_configs["config"]["set_manually"]["ego_target_speed"] = 40.0

        env_configs["config"]["set_manually"]["front_position"] = [ search_config["delta_dist"], 0.0 ]
        env_configs["config"]["set_manually"]["front_heading"] = 0.0
        env_configs["config"]["set_manually"]["front_speed"] = search_config["front_v1"]
        env_configs["config"]["set_manually"]["front_target_speed"] = search_config["front_v2"]

        # training algorithms configurations
        with open(validation_utils.configs_path + model_config_path) as f:
            model_configs = yaml.safe_load(f)
        
        # number of workers for verification tests is manually changed to zero
        model_configs["num_workers"] = 0
        
        # add environment configurations to training config
        general_config = model_configs.copy()
        general_config["env_config"] = env_configs

        print("\n[CONFIG]-> General Configurations:\t", pretty_print(general_config))
        return general_config

    def initialize_model(self, general_config: dict) -> object:
        trainer = PPOTrainer(config=general_config, env=general_config["env"]) # NOTE: change this line when model different than PPO is used
        print("\n[INFO]-> Trainer:\t", trainer)

        agent_path = os.path.join(self.repo_path, "results/trained_models/" + self.algorithm_config["load_agent_name"])
        print("\n[INFO]-> Agent Path:\t", agent_path)

        checkpoint_num = self.algorithm_config["checkpoint_number"]
        checkpoint_path = agent_path + "/checkpoint_%06i"%(checkpoint_num) + "/checkpoint-" + str(checkpoint_num)
        trainer.restore(checkpoint_path)

        print("\n[INFO]-> Restore Checkpoint:\t", checkpoint_path)
        return trainer

    def initialize_environment(self, env_configs: dict) -> Environment:
        env = Environment(config=env_configs["config"])
        
        print("\n[INFO]-> Environment:\t", env)
        return env

    def run_episode(self, env: Environment, agent: object) -> tuple:        
        statistics = {
            "ego_speeds"     :  [],
            "ego_accels"     :  [],
            "ego_jerks"      :  [],
            "ego_actions"    :  [],
            "ego_rewards"    :  [],
            "front_positions":  [],
            "front_speeds"   :  [],
            "tgap"           :  [],
            "ttc"            :  []
        }
        is_crashed = False
        episode_reward = 0.0
        
        # get initial observation
        obs = env.reset()

        # loop until episode is finished or terminated
        for step_idx in range(self.algorithm_config["max_eps_length"]):

            # get model action prediction
            action_prediction, state, _ = agent.get_policy().compute_actions([obs]) # NOTE: change this line when model different than PPO is used

            # step in the environment with predicted action to get next state
            obs, reward, done, info = env.step(action_prediction[0])
            episode_reward += reward

            # store information at every step
            statistics["ego_speeds"].append(info["ego_speed"])
            statistics["ego_accels"].append(info["ego_accel"])
            statistics["ego_jerks"].append(info["ego_jerk"])
            statistics["ego_actions"].append(info["ego_action"][0])
            statistics["ego_rewards"].append(reward)
            statistics["front_positions"].append(info["mio_position"])
            statistics["front_speeds"].append(info["mio_speed"])
            statistics["tgap"].append(info["tgap"])
            statistics["ttc"].append(info["ttc"])
            
            is_crashed = info["crashed"]
            is_terminated = info["terminated"]
            
            if done:
                if is_crashed:
                    print("\n[INFO]-> Vehicle is Crashed!")
                else:
                    print("\n[INFO]-> Episode is Finished! Length of Episode:\t", step_idx, "steps")
                break
        
        return step_idx, is_crashed, episode_reward, statistics

    def simulate(self, search_config: dict, is_tune_report: Optional[bool] = True):
        # recall trained model configurations within environment parameters
        general_config = self.initialize_config(
            env_config_path     =   "/env_config.yaml",
            search_config       =   search_config,
            model_config_path   =   "/ppo_config.yaml", # NOTE: change this line when model different than PPO is used
        )

        # create environment object with default parameters
        env = self.initialize_environment(
            env_configs         =   general_config["env_config"]
        )
        
        # load trained rl model checkpoint
        model = self.initialize_model(
            general_config      =   general_config
        )

        # run one simulation and obtain returning parameters
        episode_steps, is_crashed, total_episode_reward, statistics = self.run_episode(
                    env         =   env,
                    agent       =   model
        )

        # when ray.tune is run
        if is_tune_report:
            # report results to optimize a minimization or a maximization metric variable
            tune.report(
                crashed         =   is_crashed,
                episode_length  =   episode_steps,
                reward          =   total_episode_reward,
                statistics      =   statistics
            )
        
        # when manually statistics are required
        else:
            return statistics
