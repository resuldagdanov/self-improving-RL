import os
import yaml
import argparse
from ray import tune

repo_path = os.path.join(os.environ["BLACK_BOX"], "experiments")
configs_path = os.path.join(repo_path, "configs")


def run_search_algorithm(agent: object, validation_config: dict, seach_config: dict) -> None:
    local_directory = os.path.join(repo_path, "results/validation_checkpoints")
    save_folder_name = validation_config["experiment_name"] + "_" + validation_config["load_agent_name"] + "_Chkpt" + str(validation_config["checkpoint_number"])

    try:
        analysis = tune.run(
            run_or_experiment   =   agent.simulate,
            num_samples         =   validation_config["num_samples"],
            resources_per_trial =   validation_config["ray_tune_resources"],
            config              =   seach_config,
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