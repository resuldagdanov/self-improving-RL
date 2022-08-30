import os
import sys
import ray
import numpy as np

from collections import deque
from ray.tune.logger import pretty_print

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

from experiments.utils import validation_utils
from experiments.agents.main_agent import MainAgent
from experiments.agents.searcher import SearchAgent


class CEOptimizer:

    def __init__(self, algorithm_config: dict) -> None:
        self.algorithm_config = algorithm_config
        
        self.sample_count = 0
        self.iteration_count = 0

        sample_size = self.algorithm_config["sample_size"]
        search_space = self.algorithm_config["search_space"]
        self.sort_buffer = deque(maxlen=sample_size)

        self.min_point = [
            search_space["velocity"]["min"], # ego vehicle initial speed
            search_space["velocity"]["min"], # mio vehicle initial speed
            search_space["velocity"]["min"], # mio vehicle final speed
            search_space["distance"]["min"]  # distance between ego and mio
        ]
        self.max_point = [
            search_space["velocity"]["max"], # ego vehicle initial speed
            search_space["velocity"]["max"], # mio vehicle initial speed
            search_space["velocity"]["max"], # mio vehicle final speed
            search_space["distance"]["max"]  # distance between ego and mio
        ]

    # iterate, run, or sample and return dict
    def query(self) -> dict:
        search_configs =  self.sample_data()

        print("\n[INFO]-> Search Space:\t", pretty_print(search_configs))
        return search_configs
    
    # get uniformly sampled initial conditions
    def uniformly_sample(self) -> dict:
        return {
            "ego_v1"      : np.random.uniform(low=self.min_point[0], high=self.max_point[0]),
            "front_v1"    : np.random.uniform(low=self.min_point[1], high=self.max_point[1]),
            "front_v2"    : np.random.uniform(low=self.min_point[2], high=self.max_point[2]),
            "delta_dist"  : np.random.uniform(low=self.min_point[3], high=self.max_point[3])
        }
    
    # returns same distribution parameters for one iteration,
    # distribution parameters (min, max) will change when iteration is updated
    def sample_data(self) -> dict:
        self.sample_count += 1

        for _ in range(self.algorithm_config["check_impossible_count"]):
            # check and do not include impossible configurations
            parameters = self.uniformly_sample()
            
            # TODO: initial ego vehicle acceleration could be added to parameter space
            ego_acceleration = 0.0
            front_acceleration = np.clip(self.algorithm_config["desired_comfort_accel"] * \
                (1 - np.power(max(parameters["front_v1"], 0) / parameters["front_v2"], self.algorithm_config["velocity_exponent"])), \
                    self.algorithm_config["front_accel_range"][0], self.algorithm_config["front_accel_range"][1])
            
            if validation_utils.is_impossible_2_stop(
                initial_distance    =   parameters["delta_dist"],
                ego_velocity        =   parameters["ego_v1"],
                front_velocity      =   parameters["front_v1"],
                ego_acceleration    =   ego_acceleration, 
                front_acceleration  =   front_acceleration
            ):
                continue
            else:
                break
        
        return parameters

    # run optimizer code with result
    def update(self, config: dict, result: dict) -> None:
        # store results
        if result is not None:
            self.sort_buffer.append(result)

        # updating uniform distribution params when every iteration is finished
        if (self.sample_count != 0) and (self.sample_count % self.algorithm_config["sample_size"] == 0):
            self.cross_entropy_update()

    # updating min and max parameters of uniform distribution -> finishes one iteration
    def cross_entropy_update(self) -> None:
        self.iteration_count += 1

        # re-order list of all samples results in increasing order
        # sorting list of sample results of each discrete sample in an increasing order
        sorted_results = validation_utils.sort_samples(list_sample_results=list(self.sort_buffer), metric=self.algorithm_config["metric"])
        
        params = []
        # using better Ne examples to update min and max
        Ne = 10

        for i in range(Ne):
            params.append([
                sorted_results[i]['config/ego_v1'],
                sorted_results[i]['config/front_v1'],
                sorted_results[i]['config/front_v2'],
                sorted_results[i]['config/delta_dist']
            ])
        
        # determine new minimum and new maximum limits for uniform distribution
        self.min_point = np.amin(np.array(params), axis=0)
        self.max_point = np.amax(np.array(params), axis=0)


if __name__ == "__main__":

    # get arguments
    args = validation_utils.argument_parser()
    args.algo_config_file = "ce_search.yaml"

    # config for specified algorithm
    algorithm_config = validation_utils.get_algorithm_config(args=args)
    print("\n[CONFIG]-> Algorithm Configuration:\t", pretty_print(algorithm_config))

    # initialize main agent class
    agent = MainAgent(
        algorithm_config=algorithm_config
    )
    print("\n[INFO]-> Agent Class:\t", agent)

    # construct cross-entropy custom optimizer class
    optimizer = CEOptimizer(
        algorithm_config=algorithm_config
    )
    print("\n[INFO]-> Optimizer:\t", optimizer)

    # custom searcher class for keeping track of a metric to optimize
    searcher = SearchAgent(
        optimizer=optimizer,
        metric=algorithm_config["metric"],
        mode=algorithm_config["mode"]
    )
    print("\n[INFO]-> Searcher:\t", searcher)

    # set project directory for all ray workers
    runtime_env = {
        "working_dir": parent_directory
    }
    ray.init(runtime_env=runtime_env)

    # execute validation search algorithm and save results to csv
    validation_utils.run_search_algorithm(
        agent=agent,
        validation_config=algorithm_config,
        seach_config=None,
        search_alg=searcher
    )
