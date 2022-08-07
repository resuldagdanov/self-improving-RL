import os
import yaml
import argparse
import numpy as np
import pandas as pd

from typing import Optional
from ray import tune

repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")
configs_path = os.path.join(repo_path, "configs")


def run_search_algorithm(agent: object, validation_config: dict, seach_config: Optional[dict] = None, search_alg: Optional[object] = None) -> None:
    local_directory = os.path.join(repo_path, "results/validation_checkpoints")
    save_folder_name = validation_config["experiment_name"] + "_" + validation_config["load_agent_name"] + "_Chkpt" + str(validation_config["checkpoint_number"])

    try:
        analysis = tune.run(
            run_or_experiment   =   agent.simulate,
            num_samples         =   validation_config["num_samples"],
            resources_per_trial =   validation_config["ray_tune_resources"],
            metric              =   validation_config["metric"],
            mode                =   validation_config["mode"],
            config              =   seach_config,
            search_alg          =   search_alg,
            local_dir           =   local_directory, 
            name                =   save_folder_name,
            sync_config         =   tune.SyncConfig(),
            resume              =   validation_config["resume"]
        )

    except KeyboardInterrupt:
        print("\n[EXIT]-> Keyboard Interrupted")
        pass

    except Exception as e:
        print("\n[ERROR]-> Exception:\t", e)

    finally:
        if analysis is not None:
            experiment_data = analysis.results_df

            print("\n[INFO]-> Results Head:\t", experiment_data.head())
            print("\n[INFO]-> Results Shape:\t", experiment_data.shape)

            csv_path = os.path.join(os.path.join(local_directory, save_folder_name), "results.csv")
            experiment_data.to_csv(csv_path, index=False)
        
        else:
            print("\n[ERROR]-> Analysis is None")


def argument_parser() -> argparse:
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--algo_config_file', default="grid_search.yaml", type=str, help="yaml file path for validation experiment")
    argparser.add_argument('--resume', default=False, help="resume latest experiment: str/bool “LOCAL”, “REMOTE”, “PROMPT”, “ERRORED_ONLY”, or bool.")
    
    return argparser.parse_args()


def get_algorithm_config(args: argparse) -> dict:
    algo_yaml_path = os.path.join(os.path.join(repo_path, "configs"), args.algo_config_file)
    algo_config = yaml.load(open(algo_yaml_path), Loader=yaml.FullLoader)

    return algo_config


def sort_samples(list_sample_results: list, metric: str) -> list:
    return sorted(list_sample_results, key=lambda value: value.get(metric), reverse=False)


def extract_results_csv(file_path: str) -> pd.DataFrame:
    file_df = pd.read_csv(file_path)

    # parameters should be included in tune.report
    filtered_df = file_df[[
            "collision",
            "impossible",
            "episode_length",
            "episode_min_ttc",
            "reward",
            "statistics.ego_speeds",
            "statistics.ego_accels",
            "statistics.ego_jerks",
            "statistics.ego_actions",
            "statistics.ego_rewards",
            "statistics.front_positions",
            "statistics.front_speeds",
            "statistics.tgap",
            "statistics.ttc",
            "config.ego_v1",
            "config.front_v1",
            "config.front_v2",
            "config.delta_dist"
        ]
    ]
    return filtered_df


def is_impossible_2_stop(initial_distance: float, ego_velocity: float, front_velocity: float, ego_acceleration: float, front_acceleration: float) -> bool:
    threshold_hard_crash = 4.0 # meters (lenght of the vehicles)
    max_deceleration = 4 # TODO: get from global config
    reaction_time = 0.0 # reaction time is 0 for now, but in reality most alert driver has reation time of 1 seconds
    friction = 1.0  # TODO: include friction when dynamical model is running
    dt = 0.1 # seconds (time for forward simulation)
    
    # speeds of ego vehicle and front vehicle after reaction time is passed
    velocity_ego_think = ego_velocity + (ego_acceleration * friction * reaction_time)
    velocity_front_think = front_velocity + (front_acceleration * friction * reaction_time)

    # distance ego vehicle and front vehicle should travel while thinking time is being ellapsed
    ego_distance_think = (ego_velocity * reaction_time) + (0.5 * ego_acceleration * friction * reaction_time**2)
    front_distance_think = (front_velocity * reaction_time) + (0.5 * front_acceleration * friction * reaction_time**2)

    # total amount of ego vehicle's braking time with maximum deceleration for given friction
    brake_time = velocity_ego_think / (max_deceleration * friction)

    # current velocity of the vehicles after reation time is passed
    ego_instant_velocity = velocity_ego_think
    front_instant_velocity = velocity_front_think

    ego_instant_position = ego_distance_think
    front_instant_position = front_distance_think + initial_distance

    # when crashed during reaction time
    if abs(ego_instant_position - front_instant_position) <= threshold_hard_crash:
        return True
    
    # loop every second during braking time
    sim_steps = np.linspace(reaction_time + dt, reaction_time + brake_time + dt, int(np.ceil(brake_time / dt)))

    for i in range(len(sim_steps)):

        # distance traveled for vehicles while applying maximum deceleration
        ego_instant_velocity = max((ego_instant_velocity - (max_deceleration * dt)), 0.0)
        front_instant_velocity = max((front_instant_velocity + (front_acceleration * dt)), 0.0)
        
        ego_instant_distance = max(((ego_instant_velocity * dt) - (0.5 * max_deceleration * friction * dt**2)), 0.0)
        front_instant_distance = max(((front_instant_velocity * dt) + (0.5 * front_acceleration * friction * dt**2)), 0.0)

        # total distance all vehicles travel while ego vehicle brakes
        ego_instant_position += ego_instant_distance
        front_instant_position += front_instant_distance

        if abs(ego_instant_position - front_instant_position) <= threshold_hard_crash:
            return True
        else:
            continue
    
    return False
