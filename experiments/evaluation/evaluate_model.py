import os
import sys
import ray

from ray.tune.logger import pretty_print

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

from experiments.utils import validation_utils
from experiments.agents.main_agent import MainAgent


if __name__ == "__main__":

    # get arguments
    args = validation_utils.argument_parser()
    args.algo_config_file = "evaluation_config.yaml"
    
    # config for evaluation configurations
    eval_config = validation_utils.get_algorithm_config(args=args)
    print("\n[CONFIG]-> Evalution Configuration:\t", pretty_print(eval_config))

    # get either 'SAC' or 'PPO' others are not implemented
    agent_model_name = eval_config["load_agent_name"][:3]
    if agent_model_name == "SAC": model_config_name = "/sac_config.yaml"
    elif agent_model_name == "PPO": model_config_name = "/ppo_config.yaml"
    else: raise NotImplementedError("[ERROR]-> only SAC and PPO are implemented so far!")

    # create a folder to store the simulation results at each episode separately
    if eval_config["validation_container"] is None:
        eval_folder_name = eval_config["experiment_name"] + "_Random_Container_" + eval_config["load_agent_name"]
    else:
        eval_folder_name = eval_config["experiment_name"] + "_" + eval_config["validation_container"] + "_Container_" + eval_config["load_agent_name"]
    results_dir = os.path.join(validation_utils.repo_path, "results/evaluation_statistics")

    stats_path = os.path.join(results_dir, eval_folder_name)
    if os.path.exists(stats_path) is False:
        os.makedirs(stats_path)
    
    # initialize main agent class
    agent = MainAgent(
        algorithm_config    =   eval_config
    )
    print("\n[INFO]-> Agent Class:\t", agent)
    
    if eval_config["validation_container"] is not None:
        # get verification results dataframe
        container_df = validation_utils.extract_results_csv(
            file_path       =   os.path.join(os.path.join(parent_directory,
                                "experiments/results/validation_checkpoints"),
                                eval_config["validation_container"], "results.csv")
        )
        # sort and select scenarios with minimum time-to-collision metric
        container_df_sorted = container_df.sort_values(
            by              =   eval_config["data_sort_metric"],
            ascending       =   True
        )
        container_df_sorted = container_df_sorted[container_df_sorted["impossible"] == False] 
    else:
        # randomly sample evaluation scenarios
        if eval_config["evaluate_randomly"] is True:
            initial_conditions = {}
        else:
            # set-up initial configurations with given parameters
            initial_conditions = {
                "ego_v1"    :   eval_config["evaluation_state"]["ego_velocity"]["initial_value"],
                "front_v1"  :   eval_config["evaluation_state"]["front_velocity"]["initial_value"],
                "front_v2"  :   eval_config["evaluation_state"]["front_velocity"]["target_value"],
                "delta_dist":   eval_config["evaluation_state"]["distance"]
            }
            print("\n[INFO]-> Initial Conditions:\t", pretty_print(initial_conditions))
        
        # recall trained model configurations within environment parameters
        general_config = agent.initialize_config(
        env_config_path     =   "/env_config.yaml",
        search_config       =   initial_conditions,
        model_config_path   =   model_config_name
        )

        # create environment object with default parameters
        env = agent.initialize_environment(
            env_configs     =   general_config["env_config"]
        )
        # set seed to numpy array of environment randomness
        env.seed(eval_config["seed"])

        # load trained rl model checkpoint
        model = agent.initialize_model(
            general_config  =   general_config,
            log_folder_path =   stats_path
        )
    
    # set project directory for all ray workers
    runtime_env = {
        "working_dir"       :   parent_directory,
        "excludes"          :   ["*.err", "*.out"] # exclude error and output files (relative path from "parent_directory")
    }
    ray.shutdown()
    ray.init(runtime_env    =   runtime_env)

    # loop through each episode to evaluate
    for eps in range(eval_config["simulation_loops"]):
        print("\n[INFO]-> Episode Number:\t", eps)

        if eval_config["validation_container"] is not None:
            sample_row = container_df_sorted.iloc[eps]
            initial_conditions = {
                "ego_v1"    :   sample_row["config/ego_v1"],
                "front_v1"  :   sample_row["config/front_v1"],
                "front_v2"  :   sample_row["config/front_v2"],
                "delta_dist":   sample_row["config/delta_dist"]
            }

            # recall trained model configurations within environment parameters
            general_config = agent.initialize_config(
            env_config_path     =   "/env_config.yaml",
            search_config       =   initial_conditions,
            model_config_path   =   model_config_name
            )

            # create environment object with default parameters
            env = agent.initialize_environment(
                env_configs     =   general_config["env_config"]
            )

            # load trained rl model checkpoint
            model = agent.initialize_model(
                general_config  =   general_config,
                log_folder_path =   stats_path
            )

            env.close()

        # run one simulation (episode) loop for defined amount of steps
        statistics = agent.run_episode(
                env         =   env,
                agent       =   model
        )

        # saving statistics dictionary to folder in ./experiments/results/evaluation_statistics/
        validation_utils.save_eval_to_csv(
            stats_path      =   stats_path,
            file_name       =   "/eval_episode_" + str(eps) + ".csv",
            experiment_stats=   statistics
        )