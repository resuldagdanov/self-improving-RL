import numpy as np
import pandas as pd


class Scenarios(object):

    def __init__(self, env_config: dict) -> None:
        self.initial_scenario = env_config["set_manually"]
        self.scenario_config = env_config["scenario_config"]

    def setter(self) -> dict:
        # set default initializations when manual scenario setter type is not given
        if self.scenario_config["type"] is None:
            return self.initial_scenario
        
        else:
            return self.initial_scenario # TODO: will be changed
