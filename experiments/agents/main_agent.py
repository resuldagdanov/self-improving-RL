import os
import sys
import yaml
import numpy as np
import ray

from typing import Optional
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC

from highway_env.road.lane import AbstractLane

from highway_env.road.lane import AbstractLane

from highway_environment.envs import Environment

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

from experiments.utils import validation_utils, training_utils
from experiments.models.custom_torch_model import CustomTorchModel


class MainAgent:

    def __init__(self, algorithm_config: dict) -> None:
        self.repo_path = validation_utils.repo_path
        self.algorithm_config = algorithm_config
        
        # get either 'SAC' or 'PPO' others are not implemented
        self.agent_model_name = self.algorithm_config["load_agent_name"][:3]

        if self.agent_model_name == "SAC":
            self.model_config_name = "/sac_config.yaml"
        elif self.agent_model_name == "PPO":
            self.model_config_name = "/ppo_config.yaml"
        else:
            raise NotImplementedError("[ERROR]-> only SAC and PPO are implemented so far!")
    
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
        
        # initialize vehicles with lateral offset for corresponding lane
        lateral_offset = env_configs["config"]["initial_lane_id"] * AbstractLane.DEFAULT_WIDTH # 4m default lane width
        
        # sample random uniformly when initial conditions are not set
        if search_config == {}:
            pass
        else:
            # change vehicle initialization parameters according to search space configuration
            env_configs["config"]["set_manually"]["ego_position"] = [ 0.0, lateral_offset ]  # [x, y] -> x: longitudinal, y: lateral
            env_configs["config"]["set_manually"]["ego_heading"] = 0.0 # radians
            env_configs["config"]["set_manually"]["ego_speed"] = search_config["ego_v1"] # m/s ego vehicle initial speed
            env_configs["config"]["set_manually"]["ego_target_speed"] = 40.0 # ego vehicle target speed (is not important)

            env_configs["config"]["set_manually"]["front_position"] = [ search_config["delta_dist"], lateral_offset ]
            env_configs["config"]["set_manually"]["front_heading"] = 0.0 # radians
            env_configs["config"]["set_manually"]["front_speed"] = search_config["front_v1"]
            env_configs["config"]["set_manually"]["front_target_speed"] = search_config["front_v2"]

        # training algorithms configurations
        with open(validation_utils.configs_path + model_config_path) as f:
            model_configs = yaml.safe_load(f)
        
        # number of workers for verification tests is manually changed to zero
        model_configs["num_workers"] = 0

        # use custom nn model if required
        if self.algorithm_config["use_custom_torch_model"] is True:
            model_configs["model"]["custom_model"] = "CustomTorchModel" # TODO: change this when new model is implemented

            model_name = model_configs["model"]["custom_model"]
            if model_name == "CustomTorchModel":
                ray.rllib.models.ModelCatalog.register_custom_model(model_name, CustomTorchModel)
            else:
                print("\n[ERROR]-> Custom Model Named:\t", model_name, "is Not Supported Yet")
        
        # add environment configurations to training config
        general_config = model_configs.copy()
        general_config["env_config"] = env_configs

        print("\n[CONFIG]-> General Configurations:\t", pretty_print(general_config))
        return general_config

    def initialize_model(self, general_config: dict, log_folder_path: Optional[str] = None) -> object:
        if self.agent_model_name == "SAC":
            log_folder_name = "evaluation_SACTrainer_" + str(general_config["env"])
        elif self.agent_model_name == "PPO":
            log_folder_name = "evaluation_PPOTrainer_" + str(general_config["env"])
        else:
            raise NotImplementedError("[ERROR]-> only SAC and PPO are implemented so far (PPOTrainer or SACTrainer)!")
        
        if log_folder_path is None:
            logger_creator = None
        else:
            logger_creator = training_utils.custom_log_creator(log_folder_path, log_folder_name)
        
        if self.agent_model_name == "SAC":
            trainer = SAC(config=general_config, env=general_config["env"], logger_creator=logger_creator)
        elif self.agent_model_name == "PPO":
            trainer = PPO(config=general_config, env=general_config["env"], logger_creator=logger_creator)
        else:
            raise NotImplementedError("[ERROR]-> only SAC and PPO are implemented so far (PPOTrainer or SACTrainer)!")
        print("\n[INFO]-> Trainer:\t", trainer)

        agent_path = os.path.join(self.repo_path, "results/trained_models/" + self.algorithm_config["load_agent_name"])
        print("\n[INFO]-> Agent Path:\t", agent_path)

        checkpoint_num = self.algorithm_config["checkpoint_number"]
        checkpoint_path = agent_path + "/checkpoint_%06i"%(checkpoint_num) + "/checkpoint-" + str(checkpoint_num)
        trainer.load_checkpoint(checkpoint_path)
        # trainer.restore(checkpoint_path) # has issue with ray version 2.0.0

        print("\n[INFO]-> Restore Checkpoint:\t", checkpoint_path)
        return trainer

    def initialize_environment(self, env_configs: dict) -> Environment:
        env = Environment(config=env_configs["config"])

        if env_configs["config"]["record_video"]:
            env = validation_utils.record_video(env=env)

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
            "ttc"            :  [],
            "is_collision"   :  [],
            "is_impossible"  :  [],
            "episode_steps"  :  [],
            "episode_min_ttc":  [],
            "eps_sum_reward" :  [],
        }
        is_collision = False
        episode_reward = 0.0
        episode_min_ttc = 99.0
        
        # get initial observation
        obs = env.reset()

        # loop until episode is finished or terminated
        for step_idx in range(self.algorithm_config["max_eps_length"]):

            # get model action prediction
            action_prediction = agent.compute_single_action(obs) # NOTE: change this line when model different than PPO is used
            
            # step in the environment with predicted action to get next state
            obs, reward, done, info = env.step(action_prediction)
            episode_reward += reward

            # store information at every step
            statistics["ego_speeds"].append(info["ego_speed"])
            statistics["ego_accels"].append(info["ego_accel"])
            statistics["ego_jerks"].append(info["ego_jerk"])
            statistics["ego_actions"].append(info["ego_action"])
            statistics["ego_rewards"].append(reward)
            statistics["front_positions"].append(info["mio_position"])
            statistics["front_speeds"].append(info["mio_speed"])
            statistics["tgap"].append(info["tgap"])
            statistics["ttc"].append(info["ttc"])

            # report minimum ttc observed in this episode
            if 0.0 < info["ttc"] < episode_min_ttc:
                episode_min_ttc = info["ttc"]
            
            is_collision = info["collision"]
            is_impossible = info["impossible"]
            is_terminated = info["terminated"]

            # overwrite and store episode minimum ttc in the impossible scenario cases
            if is_impossible:
                episode_min_ttc = 99.0

            if is_impossible:
                print("\n[INFO]-> Collision is Impossible to Avoid!")
            
            if done:
                if is_collision:
                    print("\n[INFO]-> Vehicle is Crashed! Length of Episode:\t", step_idx, "steps", " Reward:", episode_reward)
                else:
                    print("\n[INFO]-> Episode is Finished! Length of Episode:\t", step_idx, "steps and Episode Reward:\t", episode_reward)
                break
        
        statistics["is_collision"] = [is_collision]
        statistics["is_impossible"] = [is_impossible]
        statistics["episode_steps"] = [step_idx]
        statistics["episode_min_ttc"] = [episode_min_ttc]
        statistics["eps_sum_reward"] = [episode_reward]

        return statistics

    def simulate(self, search_config: dict, is_tune_report: Optional[bool] = True):
        # recall trained model configurations within environment parameters
        general_config = self.initialize_config(
            env_config_path     =   "/env_config.yaml",
            search_config       =   search_config,
            model_config_path   =   self.model_config_name,
        )

        # create environment object with default parameters
        env = self.initialize_environment(
            env_configs         =   general_config["env_config"]
        )
        
        # load trained rl model checkpoint
        model = self.initialize_model(
            general_config      =   general_config,
            log_folder_path     =   None
        )

        # run one simulation and obtain returning parameters
        statistics = self.run_episode(
                    env         =   env,
                    agent       =   model
        )

        # when ray.tune is run
        if is_tune_report:
            # report results to optimize a minimization or a maximization metric variable
            tune.report(
                collision       =   statistics["is_collision"][0],
                impossible      =   statistics["is_impossible"][0],
                episode_length  =   statistics["episode_steps"][0],
                episode_min_ttc =   statistics["episode_min_ttc"][0],
                reward          =   statistics["eps_sum_reward"][0],
                statistics      =   statistics
            )
        
        # when manually statistics are required
        else:
            return statistics
