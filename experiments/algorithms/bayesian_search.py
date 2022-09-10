import os
import sys
import ray

from ray import tune
from ray.tune.logger import pretty_print
from ray.tune.search.bayesopt import BayesOptSearch

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

from experiments.utils import validation_utils
from experiments.agents.main_agent import MainAgent


if __name__ == "__main__":

    # get arguments
    args = validation_utils.argument_parser()
    args.algo_config_file = "bayesian_search.yaml"

    # config for specified algorithm
    algorithm_config = validation_utils.get_algorithm_config(args=args)
    print("\n[CONFIG]-> Algorithm Configuration:\t", pretty_print(algorithm_config))

    # initialize main agent class
    agent = MainAgent(
        algorithm_config    =       algorithm_config
    )
    print("\n[INFO]-> Agent Class:\t", agent)

    # construct search space
    distance_space, velocity_space = agent.create_search_space(
        params              =       algorithm_config['search_space']
    )
    print("\n[INFO]-> Distance Space:\t", distance_space)
    print("\n[INFO]-> Velocity Space:\t", velocity_space)

    # uniform search space configurations from given parameters
    bayes_search_space = {
        "ego_v1"            :       (velocity_space[0], velocity_space[-1]),
        "front_v1"          :       (velocity_space[0], velocity_space[-1]),
        "front_v2"          :       (velocity_space[0], velocity_space[-1]),
        "delta_dist"        :       (distance_space[0], distance_space[-1])
    }
    print("\n[INFO]-> Search Space:\t", pretty_print(bayes_search_space))

    # create build-in bayesian search algorithm object
    searcher = BayesOptSearch(
        space               =       bayes_search_space,
        metric              =       algorithm_config["metric"],
        mode                =       algorithm_config["mode"],
        random_state        =       algorithm_config["seed"],
        random_search_steps =       algorithm_config["random_search_steps"]
    )
    print("\n[INFO]-> Searcher:\t", searcher)

    # tuning configurations class -> new from ray v2.0.0
    tune_config = tune.TuneConfig(
        search_alg          =       searcher,
        num_samples         =       algorithm_config["num_samples"]
    )
    print("\n[INFO]-> TuneConfig:\t", tune_config)

    # set project directory for all ray workers
    runtime_env = {
        "working_dir"       :       parent_directory,
        "excludes"          :       ["*.err", "*.out"] # exclude error and output files (relative path from "parent_directory")
    }
    ray.init(
        runtime_env         =       runtime_env
    )

    # execute validation search algorithm and save results to csv
    validation_utils.run_search_algorithm(
        agent               =       agent,
        validation_config   =       algorithm_config,
        tune_config         =       tune_config,
        param_space         =       None
    )