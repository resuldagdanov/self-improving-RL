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
    
    # initialize main agent class
    agent = MainAgent(
        algorithm_config=eval_config
    )
    print("\n[INFO]-> Agent Class:\t", agent)

    # randomly sample evaluation scenarios
    if eval_config["evaluate_randomly"] is True:
        initial_conditions = {}
    else:
        # set-up initial configurations with given parameters
        initial_conditions = {
            'ego_v1'    : eval_config["evaluation_state"]["ego_velocity"]["initial_value"],
            'front_v1'  : eval_config["evaluation_state"]["front_velocity"]["initial_value"],
            'front_v2'  : eval_config["evaluation_state"]["front_velocity"]["target_value"],
            'delta_dist': eval_config["evaluation_state"]["distance"]
        }
        print("\n[INFO]-> Initial Conditions:\t", pretty_print(initial_conditions))

    # set project directory for all ray workers
    runtime_env = {
        "working_dir": parent_directory
    }
    ray.init(runtime_env=runtime_env)

    # recall trained model configurations within environment parameters
    general_config = agent.initialize_config(
        env_config_path     =   "/env_config.yaml",
        search_config       =   initial_conditions,
        model_config_path   =   "/ppo_config.yaml", # NOTE: change this line when model different than PPO is used
    )

    # create environment object with default parameters
    env = agent.initialize_environment(
        env_configs         =   general_config["env_config"]
    )
    
    # load trained rl model checkpoint
    model = agent.initialize_model(
        general_config      =   general_config
    )

    # set seed to numpy array of environment randomness
    env.seed(eval_config["seed"])

    for eps in range(eval_config["simulation_loops"]):
        print("\n[INFO]-> Episode Number:\t", eps)

        # run one simulation (episode) loop for defined amount of steps
        statistics = agent.run_episode(
                env         =   env,
                agent       =   model
        )

        # create a folder to store the simulation results at each episode separately
        eval_folder_name = eval_config["experiment_name"] + "_" + eval_config["load_agent_name"]

        # saving statistics dictionary to folder in ./experiments/results/evaluation_statistics/
        validation_utils.save_eval_to_csv(
            folder_name     =   eval_folder_name,
            file_name       =   "/eval_episode_" + str(eps) + ".csv",
            experiment_stats=   statistics
        )
