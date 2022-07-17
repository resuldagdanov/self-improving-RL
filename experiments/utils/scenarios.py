import sys
import numpy as np
import pandas as pd


class Scenarios(object):

    def __init__(self, env_config: dict) -> None:
        self.initial_scenario = env_config["set_manually"]
        self.scenario_config = env_config["scenario_config"]
        
        if self.scenario_config["type"] is not None:
            self.sampler = Sampler(
                scenarios_df    =   pd.read_csv(self.scenario_config["file_name"])
            )

    def setter(self) -> dict:
        # set default initializations when manual scenario setter type is not given
        if self.scenario_config["type"] is None:
            return self.initial_scenario
        
        elif self.scenario_config["type"] == "uniform":
            return self.sampler.uniform_sample_worst_30_percent()
        
        else:
            sys.exit("[ERROR]-> Defined Validation Type is Invalid!")
    
    def set_parameters(self, single_row_df: pd.DataFrame) -> dict:
        manual_scenario = {
            "ego_position"      :   [ 0.0, 0.0 ],
            "ego_heading"       :   0.0,
            "ego_speed"         :   single_row_df["config.ego_v1"].values[0],
            "ego_target_speed"  :   40.0,

            "front_position"    :   [ single_row_df["config.delta_dist"].values[0], 0.0 ],
            "front_heading"     :   0.0,
            "front_speed"       :   single_row_df["config.front_v1"].values[0],
            "front_target_speed":   single_row_df["config.front_v2"].values[0]
        }
        return manual_scenario


class Sampler(Scenarios):

    def __init__(self, scenarios_df: pd.DataFrame) -> None:
        self.scenarios_df = scenarios_df

        # total number of rows in the dataframe
        self.n_rows = len(scenarios_df.index)

        # sorting total episode reward in ascending order
        self.sort_by_reward = scenarios_df.sort_values(
            by          =   ['reward'],
            axis        =   0,
            ascending   =   True,
            inplace     =   False,  # false: returns a copy without changing original order
            ignore_index=   True    # true: returning dataframe is indexed as 0, 1, 2, ..., n-1
        )

    def uniform_sample_worst_30_percent(self) -> dict:
        """
            - uniformly samples worst total reward obtained episodes
        """

        # dropping last 70 percent of the rows
        sorted_df = self.sort_by_reward[
            self.sort_by_reward.reward < np.percentile(
                self.sort_by_reward.reward, 30)
        ]

        # uniformly select one index
        selected_idx = np.random.choice(
            a       =   np.arange(0, len(sorted_df.index), 1, dtype=int),
            size    =   1
        )

        # convert dataframe to scenario initialization configuration
        single_scenario = super().set_parameters(
            single_row_df   =   sorted_df.iloc[selected_idx]
        )
        return single_scenario
