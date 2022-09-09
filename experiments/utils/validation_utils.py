import os
import yaml
import base64
import argparse
import numpy as np
import pandas as pd

from gym.wrappers import RecordVideo
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from pathlib import Path
from typing import Optional
from ray import tune

repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")
configs_path = os.path.join(repo_path, "configs")


def run_search_algorithm(agent: object, validation_config: dict, tune_config: dict, param_space: Optional[dict] = None) -> None:
    local_directory = os.path.join(repo_path, "results/validation_checkpoints")
    save_folder_name = validation_config["experiment_name"] + "_" + validation_config["load_agent_name"] + "_Chkpt" + str(validation_config["checkpoint_number"])
    
    # create a directory to save the results
    save_folder_path = os.path.join(local_directory, save_folder_name)
    os.makedirs(save_folder_path, exist_ok=True)

    analysis = None
    try:
        tuner = tune.Tuner(
            trainable       =      agent.simulate,
            tune_config     =      tune_config,
            param_space     =      param_space,
        )
        analysis = tuner.fit()
        
    except KeyboardInterrupt:
        raise Exception("[EXIT]-> Keyboard Interrupted")
    
    finally:
        if analysis is not None:
            # get a dataframe for the last reported results of all of the trials
            experiment_data = analysis.get_dataframe()
            print("\n[INFO]-> Results Head:\t", experiment_data.head())
            print("\n[INFO]-> Results Shape:\t", experiment_data.shape)

            csv_path = os.path.join(save_folder_path, "results.csv")
            experiment_data.to_csv(csv_path, index=False)

            # report best results
            if validation_config["metric"] and validation_config["mode"] is not None:
                best_result = analysis.get_best_result(metric=validation_config["metric"], mode=validation_config["mode"])
                print("\n[INFO]-> Best Result:\t", best_result.config)
        
        else:
            raise Exception("[ERROR]-> Analysis is None")


def argument_parser() -> argparse:
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--algo_config_file', default="grid_search.yaml", type=str, help="yaml file path for validation experiment")
    argparser.add_argument('--resume', default=False, help="resume latest experiment: str/bool “LOCAL”, “REMOTE”, “PROMPT”, “ERRORED_ONLY”, or bool.")
    
    return argparser.parse_args()


def get_algorithm_config(args: argparse) -> dict:
    algo_yaml_path = os.path.join(os.path.join(repo_path, "configs"), args.algo_config_file)
    algo_config = yaml.load(open(algo_yaml_path), Loader=yaml.FullLoader)

    return algo_config


def make_float(config: dict) -> dict:
    for _, feature_name in enumerate(config): config[feature_name] = float(config[feature_name])
    return config


def sort_samples(list_sample_results: list, metric: str) -> list:
    return sorted(list_sample_results, key=lambda value: value.get(metric), reverse=False)


def save_eval_to_csv(stats_path: str, file_name: str, experiment_stats: dict) -> None:
    df = pd.DataFrame.from_dict(experiment_stats, orient="index").transpose()
    df.to_csv(stats_path + file_name)
    print("\n[INFO]-> Evaluation Stats are Saved:\t", stats_path + file_name)


def load_eval_from_csv(file_name: str) -> pd.DataFrame:
    results_dir = os.path.join(repo_path, "results/evaluation_statistics")
    stats_path = os.path.join(results_dir, file_name)
    
    if os.path.exists(stats_path) is False:
        raise Exception("[ERROR]-> File Does Not Exist in './experiments/results/evaluation_statistics': ", stats_path)
    
    stats_df = pd.read_csv(stats_path)
    print("\n[INFO]-> Evaluation Stats are Loaded:\t", stats_path)
    
    return stats_df


def extract_results_csv(file_path: str) -> pd.DataFrame:
    file_df = pd.read_csv(file_path)

    # parameters should be included in tune.report
    filtered_df = file_df[[
        "collision",
        "impossible",
        "episode_length",
        "episode_min_ttc",
        "reward",
        "statistics/ego_speeds",
        "statistics/ego_accels",
        "statistics/ego_jerks",
        "statistics/ego_actions",
        "statistics/ego_rewards",
        "statistics/front_positions",
        "statistics/front_speeds",
        "statistics/tgap",
        "statistics/ttc",
        "config/ego_v1",
        "config/front_v1",
        "config/front_v2",
        "config/delta_dist"
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


def start_video_display(visible: int = 0, size: tuple = (1400, 900)) -> Display:
    display = Display(visible=visible, size=size)
    display.start()

    return display


def record_video(env: object, video_folder: str = "videos") -> RecordVideo:
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
    # capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_video(video_path: str = "videos") -> None:
    html = []
    
    for mp4 in Path(video_path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def trajectory_extraction(dataset_path: str) -> pd.DataFrame.groupby:
    folder_path = os.path.join(repo_path, dataset_path)
    
    # making sure that real driving trajectory dataset path do exist
    if os.path.exists(folder_path) is False:
        raise Exception("[EXIT]-> Dataset Folder Does Not Exist in './experiments/dataset': ", folder_path)
    
    # get all trajectory files inside given dataset path folder
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    if len(csv_files) == 0:
        raise Exception("[EXIT]-> Real Driving Dataset Folder Does Not Contain Any Trajectories @", folder_path)
    
    trajectories = pd.DataFrame()
    
    # dataset is priorly grouped by unique IDs for each different trajectory
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        trajectories = pd.concat([trajectories, data])
    
    # grouping different trajectories by unique IDs
    grouped = trajectories.groupby("ID")

    return grouped
