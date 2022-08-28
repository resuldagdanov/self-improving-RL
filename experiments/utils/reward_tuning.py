import os
import yaml
import datetime
import numpy as np
import  validation_utils

from ray import tune
from ray.tune.logger import pretty_print

from highway_environment.envs import Environment

repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")
configs_path = os.path.join(repo_path, "configs")


class RewardTuner(object):

    def __init__(self, reward_config: dict) -> None:
        self.reward_config = reward_config

        # grouped driving scenarios based on different IDs
        self.trajectories = validation_utils.trajectory_extraction(
            dataset_path = self.reward_config["dataset_folder_path"]
        )
    
    @staticmethod
    def construct_search_space(search_space: dict) -> dict:
        # create grid search space
        velocity_multiplier_range = list(np.linspace(
            start   =   search_space["velocity_multiplier"]["min"],
            stop    =   search_space["velocity_multiplier"]["max"],
            num     =   search_space["velocity_multiplier"]["steps"],
            dtype   =   np.float32
        ))
        action_coefficient_range = list(np.linspace(
            start   =   search_space["action_coefficient"]["min"],
            stop    =   search_space["action_coefficient"]["max"],
            num     =   search_space["action_coefficient"]["steps"],
            dtype   =   np.float32
        ))
        timegap_coefficient_range = list(np.linspace(
            start   =   search_space["timegap_coefficient"]["min"],
            stop    =   search_space["timegap_coefficient"]["max"],
            num     =   search_space["timegap_coefficient"]["steps"],
            dtype   =   np.float32
        ))
        ttc_multiplier_range = list(np.linspace(
            start   =   search_space["ttc_multiplier"]["min"],
            stop    =   search_space["ttc_multiplier"]["max"],
            num     =   search_space["ttc_multiplier"]["steps"],
            dtype   =   np.float32
        ))

        space_config = {
            "rew_speed_coef"    :   tune.grid_search(velocity_multiplier_range),
            "rew_u_coef"       :   tune.grid_search(action_coefficient_range),
            "rew_tgap_coef"     :   tune.grid_search(timegap_coefficient_range),
            "rew_ttc_coef"      :   tune.grid_search(ttc_multiplier_range)
        }
        return space_config
    
    def compute_statistics(self, searching_config: dict) -> dict:
        statistics = {}

        # update reward parameters of the reward function
        env.config.update(searching_config)

        idx = 0
        # loop through different ACC scenarios
        for _name, trajectory in self.trajectories:
            sum_reward = 0.0

            stats = {
                "avg_rewards"       :   0.0,
                "ego_velocity"      :   trajectory["ego_speed"].tolist(),
                "ego_acceleration"  :   trajectory["a_x"].tolist(),
                "ego_jerk"          :   trajectory["jerk"].tolist(),
                "throttle_position" :   trajectory["throttle_pos"].tolist(),
                "brake_position"    :   trajectory["brake_pos"].tolist(),
                "mio_position"      :   trajectory["mio_pos"].tolist(),
                "mio_velocity"      :   trajectory["mio_vel"].tolist(),
                "mio_id"            :   trajectory["mio_id"].iloc[0],
                "ttc"               :   trajectory["ttc"].tolist(),
            }

            # loop through each row step in selected trajectory to calculate reward at each step
            for step in range(trajectory.shape[0]):

                action = stats["throttle_position"][step] - stats["brake_position"][step]
                raw_obs = {
                    "ego_speed"     :   stats["ego_velocity"][step],
                    "mio_pos"       :   stats["mio_position"][step],
                    "mio_vel"       :   stats["mio_velocity"][step],
                    "ego_jerk"      :   stats["ego_jerk"][step]
                }
                
                reward = env.compute_reward(
                    obs     =   raw_obs,
                    action  =   [action]
                )
                sum_reward += reward
            
            stats["avg_rewards"] = sum_reward / (step + 1)
            statistics[idx] = stats
            
            idx += 1
            if idx == self.reward_config["num_tune_scenarios"]:
                break
        
        return statistics
    
    def run_tuner(self, config):
        # compute scenarios episode statistics from given trajectories
        statistics = self.compute_statistics(
            searching_config= config
        )

        # maximize reward from the result of the simulation
        tune.report(
            reward          =   np.mean([stats["avg_rewards"] for stats in statistics.values()]),
            stats           =   statistics
        )


if __name__ == "__main__":

    # get arguments
    args = validation_utils.argument_parser()
    args.algo_config_file = "reward_tuning.yaml"

    # config for tuning reward parameters
    reward_config = validation_utils.get_algorithm_config(args=args)
    print("\n[CONFIG]-> Reward Tuner Configuration:\t", pretty_print(reward_config))

    # highway environment configirations
    with open(validation_utils.configs_path + "/env_config.yaml") as f:
        env_config = yaml.safe_load(f)
    
    # environment object is create to use custom reward calculation function
    env = Environment(
        config = env_config["config"]
    )
    print("\n[INFO]-> Environment:\t", env)
    
    reward_tuner = RewardTuner(
        reward_config = reward_config
    )
    print("\n[INFO]-> Tune Reward Function:\t", reward_tuner)

    # search space configuration
    searching_config = reward_tuner.construct_search_space(
        search_space = reward_config["search_space"]
    )

    local_directory = os.path.join(repo_path, "results/tuning_reward_function")
    save_folder_name = reward_config["experiment_name"] + "_" + datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    print("\n[INFO]-> Running Results Path:\t", local_directory + "/" + save_folder_name)

    try:
        analysis = tune.run(
            run_or_experiment   =   reward_tuner.run_tuner,
            num_samples         =   reward_config["num_samples"],
            resources_per_trial =   reward_config["ray_tune_resources"],
            metric              =   reward_config["metric"],
            mode                =   reward_config["mode"],
            config              =   searching_config,
            search_alg          =   None,
            local_dir           =   local_directory, 
            name                =   save_folder_name,
            sync_config         =   tune.SyncConfig(),
            resume              =   reward_config["resume"]
        )
    except KeyboardInterrupt:
        raise Exception("[EXIT]-> Keyboard Interrupted")

    except Exception as e:
        raise Exception("[ERROR]-> Exception Occured:\t", e)

    finally:
        if analysis is not None:
            experiment_data = analysis.results_df

            print("\n[INFO]-> Results Head:\t", experiment_data.head())
            print("\n[INFO]-> Results Shape:\t", experiment_data.shape)

            csv_path = os.path.join(os.path.join(local_directory, save_folder_name), "results.csv")
            experiment_data.to_csv(csv_path, index=False)
        
        else:
            raise Exception("[ERROR]-> Analysis is None")
