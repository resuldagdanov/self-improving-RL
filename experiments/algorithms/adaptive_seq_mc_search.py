import os
import sys
import ray
import copy
import numpy as np

from ray.tune.logger import pretty_print

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

from experiments.utils import validation_utils
from experiments.agents.main_agent import MainAgent
from experiments.agents.searcher import SearchAgent


class AdapSeqMCOptimizer:

    def __init__(self, algorithm_config: dict) -> None:
        self.algorithm_config = algorithm_config
        
        self.noise_level = self.algorithm_config["noise_level"]
        self.metric = self.algorithm_config["metric"]
        
        self.J = self.algorithm_config["J"]
        self.Ne = self.algorithm_config["Ne"]
        self.K = int(0.2 * self.Ne)

        search_space = self.algorithm_config["search_space"]

        counter = 0
        self.trial_counter = 0
        self.Fj_counter = 0
        self.result_buffer = []
        self.current_trials = []
        self.update_ongoing = False

        if self.algorithm_config["space_type"] == "log":
            self.Lj = np.logspace(
                start=np.log10(self.algorithm_config["eps_init"]),
                stop=np.log10(self.algorithm_config["eps_final"]),
                num=self.J
            )
        elif self.algorithm_config["space_type"] == "lin":
            self.Lj = np.linspace(
                start=self.algorithm_config["eps_init"],
                stop=self.algorithm_config["eps_final"],
                num=self.J
            )
        else:
            raise ValueError("[ERROR]-> Invalid Space Type (must be 'log' or 'lin')")
        
        print("\n[INFO]-> Search Space:\t", self.Lj)

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

        # check and do not include impossible configurations
        while counter < self.Ne:
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
                counter += 1
                self.current_trials.append(copy.deepcopy(parameters))
        
        self.Fj = [self.current_trials]

    # iterate, run, or sample and return dict
    def query(self) -> dict:
        if self.Fj_counter != self.J:
            
            if len(self.result_buffer) < self.Ne and self.trial_counter < self.Ne:
                self.trial_counter += 1

                search_configs = validation_utils.make_float(self.current_trials[self.trial_counter])
                print("\n[INFO]-> Search Space:\t", pretty_print(search_configs))

                return search_configs
            else:
                return None
            
        else:
            return "[INFO]-> FINISHED!"
    
    # get uniformly sampled initial conditions
    def uniformly_sample(self) -> dict:
        return {
            "ego_v1"      : np.random.uniform(low=self.min_point[0], high=self.max_point[0]),
            "front_v1"    : np.random.uniform(low=self.min_point[1], high=self.max_point[1]),
            "front_v2"    : np.random.uniform(low=self.min_point[2], high=self.max_point[2]),
            "delta_dist"  : np.random.uniform(low=self.min_point[3], high=self.max_point[3])
        }
    
    # add uniform noise to samples to be selected
    def apply_noise(self, params: dict, noise_level: float = 0.5) -> dict:
        for _, feature in enumerate(params):

            noise = np.random.uniform(low=(-1) * noise_level, high=noise_level)
            params[feature] += noise
            
            if feature == "ego_v1":
                params[feature] = np.clip(a=params[feature], a_min=self.min_point[0], a_max=self.max_point[0])
            elif feature == "front_v1":
                params[feature] = np.clip(a=params[feature], a_min=self.min_point[1], a_max=self.max_point[1])
            elif feature == "front_v2":
                params[feature] = np.clip(a=params[feature], a_min=self.min_point[2], a_max=self.max_point[2])
            elif feature == "delta_dist":
                params[feature] = np.clip(a=params[feature], a_min=self.min_point[3], a_max=self.max_point[3])
            else:
                raise ValueError("[ERROR]-> Invalid Sample Feature Name")
        
        return params

    # run optimizer code with result
    def update(self, config: dict, result: dict) -> None:

        if self.update_ongoing is False and self.Fj_counter < self.J:
            self.result_buffer.append([config, result])
            
            if len(self.result_buffer) >= self.Ne:
                self.update_ongoing = True
                
                sampling_buffer, new_trials, results = [], [], []

                for trial_config, trial_result in self.result_buffer:
                    results.append(trial_result[self.metric])
                sorted_results_indices = np.argsort(results)
                
                for k in range(self.K):
                    sampling_buffer.append(self.result_buffer[sorted_results_indices[k]][0])
                
                while sampling_buffer == []:
                    if self.Fj_counter <= 0:
                        self.Fj_counter =  self.J
                        return 
                    
                    self.Fj_counter -= 1
                    
                    for trial_config, trial_result in self.result_buffer:
                        if trial_result[self.metric] <= self.Lj[self.Fj_counter]:
                            sampling_buffer.append(trial_config)
                
                # check and do not include impossible configurations
                for _ in range(self.Ne):
                    sampled_config = np.random.choice(a=sampling_buffer)
                    
                    parameters = self.apply_noise(
                        params=copy.deepcopy(sampled_config),
                        noise_level=self.noise_level
                    )

                    # loop until 'possible to avoid collision' configuration is found
                    for _ in range(self.algorithm_config["check_impossible_count"]):
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
                    
                    new_trials.append(parameters)
                
                self.trial_counter = 0
                self.result_buffer = []
                self.current_trials = copy.deepcopy(new_trials)

                self.Fj_counter += 1
                self.Fj.append(copy.deepcopy(self.current_trials))

                self.update_ongoing = False


if __name__ == "__main__":

    # get arguments
    args = validation_utils.argument_parser()
    args.algo_config_file = "adaptive_seq_mc.yaml"

    # config for specified algorithm
    algorithm_config = validation_utils.get_algorithm_config(args=args)
    print("\n[CONFIG]-> Algorithm Configuration:\t", pretty_print(algorithm_config))

    # initialize main agent class
    agent = MainAgent(
        algorithm_config=algorithm_config
    )
    print("\n[INFO]-> Agent Class:\t", agent)

    # construct adaptive sequential monte carlo optimizer class
    optimizer = AdapSeqMCOptimizer(
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
