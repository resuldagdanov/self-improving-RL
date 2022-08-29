import sys
import numpy as np
import pandas as pd


class Scenarios(object):

    def __init__(self, env_config: dict) -> None:
        self.initial_scenario = env_config["set_manually"]
        self.scenario_config = env_config["scenario_config"]
        
        if self.scenario_config["type"] is not None:
            self.sampler = Sampler(
                scenarios_list    =   self.scenario_config["file_name"]
            )

    def setter(self) -> dict:
        # set default initializations when manual scenario setter type is not given
        if self.scenario_config["type"] is None:
            return self.initial_scenario
        
        elif self.scenario_config["type"] == "uniform":
            return self.sampler.uniform_sample_worst_30_percent()
        
        elif self.scenario_config["type"] == "mixture":
            return self.sampler.mixture_sample_with_percentage()
        
        else:
            sys.exit("[ERROR]-> Defined Validation Type is Invalid!")
    
    def set_parameters(self, single_row_df: pd.DataFrame) -> dict:
        manual_scenario = {
            "ego_position"      :   [ 0.0, 0.0 ],
            "ego_heading"       :   0.0,
            "ego_speed"         :   single_row_df["config/ego_v1"].values[0],
            "ego_target_speed"  :   40.0,

            "front_position"    :   [ single_row_df["config/delta_dist"].values[0], 0.0 ],
            "front_heading"     :   0.0,
            "front_speed"       :   single_row_df["config/front_v1"].values[0],
            "front_target_speed":   single_row_df["config/front_v2"].values[0]
        }
        return manual_scenario


class Sampler(Scenarios):

    def __init__(self, scenarios_list: list) -> None:
        self.scenarios_dfs = []
        self.probabilities = []
        self.sort_df_by_reward = []

        # get all verification results path and probabiliry percentage
        for tuple_result in scenarios_list:
            result_csv_path, prabability = tuple_result
            scenario_df = pd.read_csv(result_csv_path)

            # sorting total episode reward in ascending order
            self.sort_df_by_reward.append(scenario_df.sort_values(
                by          =   ["reward"],
                axis        =   0,
                ascending   =   True,
                inplace     =   False,  # false: returns a copy without changing original order
                ignore_index=   True    # true: returning dataframe is indexed as 0, 1, 2, ..., n-1
                )
            )
            # total number of rows in the dataframe
            n_rows = len(scenario_df.index)

            self.scenarios_dfs.append(scenario_df)
            self.probabilities.append(prabability)
        
        if (1 - 1e-8) < sum(self.probabilities) < (1 + 1e-8):
            pass
        else:
            sys.exit("[ERROR]-> Scenario Selection Probabilities Do Not Sum Up To 1!")
    
    def uniform_sample_worst_30_percent(self) -> dict:
        """
            - uniformly samples worst total reward obtained episodes
        """
        sorted_df = self.sort_df_by_reward[0]

        # dropping last 70 percent of the rows
        filtered_df = sorted_df[
            sorted_df.reward < np.percentile( sorted_df.reward, 30)
        ]

        # uniformly select one index
        selected_idx = np.random.choice(
                    a       =   np.arange(0, len(filtered_df.index), 1, dtype=int),
                    size    =   1
        )

        # convert dataframe to scenario initialization configuration
        single_scenario = super().set_parameters(
            single_row_df   =   filtered_df.iloc[selected_idx]
        )
        return single_scenario
    
    def mixture_sample_with_percentage(self) -> dict:
        """
            - sample from given verification scenarios with predefined random sampling precentage
        """
        samples_to_choose = pd.DataFrame()

        # get one uniformly random sample from all verification results
        for scenarios in self.scenarios_dfs:
            # uniformly select one index
            choosen_idx = np.random.choice(
                    a       =   np.arange(0, len(scenarios.index), 1, dtype=int),
                    size    =   1
            )
            
            samples_to_choose = pd.concat(
                objs        =   [samples_to_choose, scenarios.iloc[choosen_idx]],
                axis        =   0,
                ignore_index=   False
            )
        
        # select one index out of combined dataframe with given probabilities
        selected_idx = np.random.choice(
                    a       =   np.arange(0, len(samples_to_choose), 1, dtype=int),
                    size    =   1,
                    p       =   self.probabilities
        )

        # convert dataframe to scenario initialization configuration
        single_scenario = super().set_parameters(
            single_row_df   =   samples_to_choose.iloc[selected_idx]
        )
        return single_scenario
    