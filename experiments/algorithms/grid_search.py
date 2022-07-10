import os
import sys
import ray

from ray import tune
from ray.tune.logger import pretty_print

sys.path.append(os.path.join(os.environ["BLACK_BOX"]))

from experiments.utils import validation_utils
from experiments.agents.main_agent import MainAgent


if __name__ == "__main__":

    # get arguments
    args = validation_utils.argument_parser()
    args.algo_config_file = "grid_search.yaml"

    # config for specified algorithm
    algorithm_config = validation_utils.get_algorithm_config(args=args)
    print("\n[CONFIG]-> Algorithm Configuration:\t", pretty_print(algorithm_config))

    # initialize main agent class
    agent = MainAgent(
        algorithm_config=algorithm_config
    )
    print("\n[INFO]-> Agent Class:\t", agent)

    # construct search space
    distance_space, velocity_space = agent.create_search_space(
        params=algorithm_config['search_space']
    )
    print("\n[INFO]-> Distance Space:\t", distance_space)
    print("\n[INFO]-> Velocity Space:\t", velocity_space)

    ray.init()

    # gridding search space configurations from given parameters
    search_configs = {
        'ego_v1'    : tune.grid_search(velocity_space),
        'front_v1'  : tune.grid_search(velocity_space),
        'front_v2'  : tune.grid_search(velocity_space),
        'delta_dist': tune.grid_search(distance_space)
    }
    print("\n[INFO]-> Search Space:\t", pretty_print(search_configs))

    # execute validation search algorithm and save results to csv
    validation_utils.run_search_algorithm(
        agent=agent,
        validation_config=algorithm_config,
        seach_config=search_configs
    )