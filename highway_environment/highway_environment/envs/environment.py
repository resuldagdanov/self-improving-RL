import os
import sys
import yaml
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import action_factory, Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle

from typing import Tuple, Optional
from highway_environment.envs.observation import Observation

parent_directory = os.path.join(os.environ["BLACK_BOX"])
sys.path.append(parent_directory)

from experiments.utils import scenarios, validation_utils


class Environment(AbstractEnv):
    
    with open(os.path.join(os.environ["BLACK_BOX"], "highway_environment/highway_environment/default_configs/env_config.yaml")) as f:
        init_config = yaml.safe_load(f)
    
    def __init__(self, config: Optional[dict] = init_config["config"]) -> None:
        self.scenario_runner = scenarios.Scenarios(
            env_config=config
        )
        super(Environment, self).__init__(config)

        self.rendering_mode = self.config["rendering_mode"]
        self.enable_auto_render = True
    
    @classmethod
    def default_config(cls) -> dict:
        return super().default_config()
    
    def define_spaces(self) -> None:
        self.action_type = action_factory(self, self.config["action"])
        self.action_space = self.action_type.space()

        self.observation_type = Observation(
            env=self.action_type.env,
            frequency=self.config["policy_frequency"],
            clip=self.action_type.clip
        )
        self.observation_space = self.observation_type.space()
    
    def _reset(self) -> None:
        # define lanes and road profile
        self._create_road()

        # whether to set scenario from predefined configurations
        self.config["set_manually"] = self.scenario_runner.setter()
        
        # spawn all vehicles
        self._create_vehicles()

        if self.config["rendering"]:
            self.action_type.env.render()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
            np_random=self.np_random, record_history=self.config["show_trajectories"]
        )

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # update constant parameters for IDM
        other_vehicles_type.ACC_MAX = self.config["idm_max_accel"]
        other_vehicles_type.COMFORT_ACC_MAX = self.config["idm_comfort_acc_max"]
        other_vehicles_type.COMFORT_ACC_MIN = self.config["idm_comfort_acc_min"]
        other_vehicles_type.DISTANCE_WANTED = self.config["idm_distance_wanted"] + ControlledVehicle.LENGTH
        other_vehicles_type.TIME_WANTED = self.config["idm_time_wanted"]
        other_vehicles_type.DELTA = self.config["idm_delta"]

        if self.config["controlled_vehicles"] != 0:
            other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        else:
            other_per_controlled = range(1, self.config["vehicles_count"]) # TODO: does not work work with vehicles_count=1
        
        # reset impossibility condition of avoiding collision
        self.is_impossible = False

        self.controlled_vehicles = []
        for others in other_per_controlled:

            # manually set ego vehicle position and speed for verification algorithms
            if len(self.config["set_manually"]) != 0:
                # set IDM for ego vehicle when number of controlled vehicles is 0
                if self.config["controlled_vehicles"] == 0:
                    vehicle = IDMVehicle(
                        road=self.road,
                        position=self.config["set_manually"]["ego_position"],
                        heading=self.config["set_manually"]["ego_heading"],
                        speed=self.config["set_manually"]["ego_speed"],
                        target_speed=self.config["set_manually"]["ego_target_speed"],
                        target_lane_index=("0", "1", self.config["initial_lane_id"])
                    )
                    controlled_vehicle = other_vehicles_type.create_from(vehicle=vehicle)
                else:
                    vehicle = ControlledVehicle(
                        road=self.road,
                        position=self.config["set_manually"]["ego_position"],
                        heading=self.config["set_manually"]["ego_heading"],
                        speed=self.config["set_manually"]["ego_speed"],
                        target_speed=self.config["set_manually"]["ego_target_speed"],
                        target_lane_index=None
                    )
                    controlled_vehicle = self.action_type.vehicle_class.create_from(vehicle=vehicle)
            
            # randomly set ego vehicle position and initial velocity
            else:
                ego_speed = self.np_random.normal(np.random.uniform(*(self.config["speed_range"])), self.config["speed_std"])
                
                # set IDM for ego vehicle when number of controlled vehicles is 0
                if self.config["controlled_vehicles"] == 0:
                    controlled_vehicle = other_vehicles_type.create_random(
                        road=self.road,
                        speed=ego_speed,
                        lane_id=self.config["initial_lane_id"]
                    )
                    controlled_vehicle.target_speed = self.config["speed_range"][-1]
                else:
                    controlled_vehicle = self.action_type.vehicle_class.create_random(
                        road=self.road,
                        speed=ego_speed,
                        lane_id=self.config["initial_lane_id"]
                    )
            
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for idx in range(others):
                # manually set other vehicle position and speed for verification algorithms
                if len(self.config["set_manually"]) != 0:
                    # TODO: should be inside search space
                    lane_index = ("0", "1", self.config["initial_lane_id"]) if idx == 0 else None
                    
                    vehicle = ControlledVehicle(
                        road=self.road,
                        position=self.config["set_manually"]["front_position"],
                        heading=self.config["set_manually"]["front_heading"],
                        speed=self.config["set_manually"]["front_speed"],
                        target_speed=self.config["set_manually"]["front_target_speed"],
                        target_lane_index=lane_index
                    )
                    other_vehicle = other_vehicles_type.create_from(vehicle=vehicle)
                
                # randomly set other vehicle position and initial velocity
                else:
                    # assign lane id same with the ego vehicle's for the first mio vehicle
                    lane_id = self.config["initial_lane_id"] if idx == 0 else None

                    # randomly select initial conditions
                    tgap = self.np_random.normal(self.config["tgap_mean"], self.config["tgap_std"])
                    tgap = np.clip(tgap, self.config["min_tgap"], self.config["max_tgap"])
                    speed = self.np_random.normal(ego_speed, self.config["speed_std"])
                    
                    # spawn other vehicle
                    other_vehicle = other_vehicles_type.create_random(
                        road=self.road,
                        spacing=tgap,
                        speed=speed, # other vehicle velocity in m/s
                        lane_from=None, # start node of the lane to spawn in
                        lane_to=None,   # end node of the lane to spawn in
                        lane_id=lane_id # id of the lane to spawn in
                    )
                    target_speed = self.np_random.normal(speed, self.config["speed_std"])
                    other_vehicle.target_speed = target_speed
                
                self.road.vehicles.append(other_vehicle)

                # make extra first action to compute an initial acceleration of the other vehicle
                other_vehicle.act()
                
                # currently only checks impossible to avoid collision scenarios with only front mio vehicles
                self.is_impossible = validation_utils.is_impossible_2_stop(
                    initial_distance=other_vehicle.position[0] - controlled_vehicle.position[0],
                    ego_velocity=controlled_vehicle.speed,
                    front_velocity=other_vehicle.speed,
                    ego_acceleration=controlled_vehicle.action["acceleration"],
                    front_acceleration=other_vehicle.action["acceleration"]
                )
    
    def _is_terminal(self) -> bool:
        return self.vehicle.crashed or self.is_impossible or \
            self.steps >= self.config["episode_length"] or \
                self.config["offroad_terminal"] and not self.vehicle.on_road

    def _info(self, obs: np.ndarray, action: Action, reward: float, terminated: bool) -> dict:
        info = {
            "obs": list(obs),
            "ego_action": action,
            "ego_speed": self.vehicle.speed,
            "ego_accel": self.observation_type.raw_obs["ego_accel"],
            "ego_jerk": self.observation_type.raw_obs["ego_jerk"],
            "mio_position": self.observation_type.raw_obs["mio_pos"],
            "mio_speed": self.observation_type.raw_obs["mio_vel"],
            "tgap": self.observation_type.raw_obs["tgap"],
            "ttc": self.observation_type.raw_obs["ttc"],
            "reward": reward,
            "collision": self.vehicle.crashed,
            "impossible": self.is_impossible,
            "terminated": terminated
        }

        return info
    
    def _simulate(self, action: Optional[Action] = None) -> None:
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        
        for frame in range(frames):
            # forward action to the vehicle
            if action is not None \
                and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1

            # automatically render intermediate simulation steps if a viewer has been launched
            # ignored if the rendering is done offscreen
            if frame < frames - 1:  # last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        # true: render each step; false: render only initial state
        self.enable_auto_render = True
    
    def _reward(self, action: Action) -> float:
        # get observation dictionary elements without normalizations
        obs = self.observation_type.raw_obs

        # desired speed reward
        speed_ratio = min(1, obs["ego_speed"] / self.config["speed_range"][1])
        speed_rew = speed_ratio * self.config["rew_speed_coef"]
        
        # calculate time-gap in seconds
        tgap = np.clip(obs["mio_pos"] / (obs["ego_speed"] + 1e-5), 0, self.config["max_tgap"])
        
        # time-gap punishment
        if 0 < tgap < self.config["rew_tgap_range"][0]:
            tgap_rew = max(-1 / tgap, -10)
        elif tgap > self.config["rew_tgap_range"][1]:
            tgap_rew = max(-tgap, -10)
        else:
            tgap_rew = tgap
        
        # collision punishment or time-gap and speed reward jerk punishment
        reward = self.config["collision_reward"] if self.vehicle.crashed else \
            (self.config["rew_tgap_coef"] * tgap_rew) + speed_rew + (abs(obs["ego_jerk"]) * self.config["rew_jerk_coef"])
            
        return reward

    def default_reward(self, action: Action) -> float:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]
        
        # use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["speed_range"], [0, 1])
        
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        
        reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"] + self.config["right_lane_reward"]], [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        
        return reward

    def custom_reward(self, action: Action) -> float:
        # get observation dictionary elements without normalizations
        obs = self.observation_type.raw_obs

        # split action into steering and throttle-brake
        if self.config["action"]["lateral"] and self.config["action"]["longitudinal"]:
            accel_action, steer_action = action
        elif self.config["action"]["lateral"]:
            accel_action, steer_action = 0.0, action[0]
        elif self.config["action"]["longitudinal"]:
            accel_action, steer_action = action[0], 0.0
        else:
            accel_action, steer_action = 0.0, 0.0
        
        # desired speed reward
        speed_ratio = min(1, obs["ego_speed"] / self.config["speed_range"][1])
        speed_rew = speed_ratio * self.config["rew_speed_coef"]
        
        # too slow ego vehicle punishment
        if obs["ego_speed"] < 1.0:
            too_slow = -0.5
        else:
            too_slow = 0.0
        
        # calculate time-gap and time-to-collision with reward function configuration limits
        tgap = np.clip(obs["mio_pos"] / (obs["ego_speed"] + 1e-5), 0, self.config["max_tgap"])
        ttc = -obs["mio_pos"] / (obs["mio_vel"] - 1e-5) if obs["mio_vel"] < 0 else self.config["max_ttc"]
        
        # time-gap punishment
        if 0 < tgap < self.config["rew_tgap_range"][0]:
            tgap_rew = max(-1 / tgap, -10) * self.config["rew_tgap_coef"]
            speed_rew = 0.0
        elif tgap > self.config["rew_tgap_range"][1]:
            tgap_rew = max(-tgap, -10) * self.config["rew_tgap_coef"] / 4
        else:
            tgap_rew = 0.0
        
        # time-to-collision punishment
        if ttc < self.config["min_ttc"]:
            ttc_rew = ((ttc - self.config["min_ttc"]) / self.config["min_ttc"]) * self.config["rew_ttc_coef"]
        else:
            ttc_rew = 0.0
        
        # input action cost
        if not (self.config["rew_u_range"][0] < action[0] < self.config["rew_u_range"][1]):
            eco_rew = -abs(action[0]) * self.config["rew_u_coef"]
        else:
            eco_rew = abs(accel_action) * self.config["rew_u_coef"]
        
        # input steering action cost
        steer_rew = abs(steer_action) * self.config["rew_steer_coef"]

        # jerk punishment
        if abs(obs["ego_jerk"]) > self.config["rew_jerk_lim"]:
            jerk_rew = abs(obs["ego_jerk"]) * self.config["rew_jerk_coef"]
        else:
            jerk_rew = 0.0
        
        # collision punishment
        if self.vehicle.crashed:
            collision_rew = self.config["collision_reward"]
        else:
            collision_rew = 0.0
        
        # out of track punishment
        if not self.vehicle.on_road:
            track_rew = self.config["offroad_reward"]
        else:
            track_rew = 0.0
        
        reward = float(speed_rew + tgap_rew + ttc_rew + eco_rew + steer_rew + jerk_rew + too_slow + collision_rew + track_rew)
        
        return reward
    
    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action, reward, terminal)

        return obs, reward, terminal, info