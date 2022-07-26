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

    # set-up initial configurations with given parameters
    initial_conditions = {
        'ego_v1'    : eval_config["evaluation_state"]["velocity"]["min"],
        'front_v1'  : eval_config["evaluation_state"]["velocity"]["min"],
        'front_v2'  : eval_config["evaluation_state"]["velocity"]["max"],
        'delta_dist': eval_config["evaluation_state"]["distance"]
    }
    print("\n[INFO]-> Initial Conditions:\t", pretty_print(initial_conditions))

    # set project directory for all ray workers
    runtime_env = {
        "working_dir": parent_directory
    }
    ray.init(runtime_env=runtime_env)
    
    for eps in range(eval_config["simulation_loops"]):
        # run one simulation (episode) loop for defined amount of steps
        statistics = agent.simulate(search_config=initial_conditions, is_tune_report=False)
