import yaml
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import action_factory, Action
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split

from typing import Tuple, Optional
from highway_environment.src.observation import Observation


class Environment(AbstractEnv):

    with open(f"./default_configs/env_config.yaml") as f:
        init_config = yaml.safe_load(f)
    
    def __init__(self, config=init_config):
        super(Environment, self).__init__(config)
        
        self.rendering_mode = self.config["rendering_mode"]
        self.enable_auto_render = True
        self.steps = 0  # actions performed

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "initial_lane_id": None # change this to start agent from spesific lane id
        })
        
        return config

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
        self._create_road()
        self._create_vehicles()

        if not self.config["rendering"]:
            self.viewer = None
        else:
            self.viewer = EnvViewer(
                env=self.action_type.env,
                config=self.config
            )

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
            np_random=self.np_random, record_history=self.config["show_trajectories"]
        )

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []

        for others in other_per_controlled:
            ego_speed = self.np_random.normal(np.random.uniform(*(self.config["speed_range"])), self.config["speed_std"])
            
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=ego_speed,
                lane_id=self.config["initial_lane_id"]
            )
            
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                tgap = self.np_random.normal(self.config["tgap_mean"], self.config["tgap_std"])
                tgap = np.clip(tgap, self.config["min_tgap"], self.config["max_tgap"])
                speed = self.np_random.normal(ego_speed, self.config["speed_std"])
                v = other_vehicles_type.create_random(self.road, spacing=tgap,speed=speed)
                target_speed = self.np_random.normal(speed, self.config["speed_std"])
                v.target_speed = target_speed

                self.road.vehicles.append(v)

    def _is_terminal(self) -> bool:
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _info(self, obs: np.ndarray, action: Action, reward: float, terminated: bool) -> dict:
        info = {
            "obs": list(obs),
            "action": list(action),
            "reward": round(reward, 4),
            "speed": round(self.vehicle.speed, 4),
            "accel": round(self.observation_type.raw_obs['ego_accel'], 4),
            "jerk": round(self.observation_type.raw_obs['ego_jerk'], 4),
            "crashed": self.vehicle.crashed,
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
        raw_obs = self.observation_type.raw_obs

        # calculate time-gap and time-to-collision
        tgap = np.clip(raw_obs["mio_pos"] / (raw_obs["ego_speed"] + 1e-5), 0, self.config["max_tgap"])
        ttc = - raw_obs["mio_pos"] / (raw_obs["mio_vel"] - 1e-5) if raw_obs["mio_vel"] < 0 else self.config["max_ttc"]
        # ttc = np.clip(ttc, 0, self.config["max_ttc"])

        # desired speed reward
        speed_ratio = min(1, raw_obs["ego_speed"] / self.config["speed_range"][1])
        speed_rew = speed_ratio * self.config["rew_speed_coef"]
        
        # too slow ego vehicle punishment
        if raw_obs["ego_speed"] < 1:
            too_slow = - 0.5
        else:
            too_slow = 0.0
        
        # time-gap punishment
        if 0 < tgap < self.config["rew_tgap_range"][0]:
            tgap_rew = max(-1 * (1 / tgap), -10) * self.config["rew_tgap_coef"]
            speed_rew = 0
        elif tgap > self.config["rew_tgap_range"][1]:
            tgap_rew = max(-tgap, -10) * self.config["rew_tgap_coef"] / 4
        else:
            tgap_rew = 0.0
        
        # time-to-collision reward
        if ttc < self.config["min_ttc"]:
            ttc_rew = (ttc - self.config["min_ttc"]) / self.config["min_ttc"] * self.config["rew_ttc_coef"]
        else:
            ttc_rew = 0.0
        
        # input action cost
        if not (self.config["rew_u_range"][0] < action[0] < self.config["rew_u_range"][1]):
            eco_rew = - abs(action[0]) * self.config["rew_u_coefs"][1]
        else:
            eco_rew = - abs(action[0]) * self.config["rew_u_coefs"][0]
        
        # jerk punishment
        if abs(raw_obs["ego_jerk"]) > self.config["rew_jerk_lim"]:
            jerk_rew = - (abs(raw_obs["ego_jerk"]) / 20) * self.config["rew_jerk_coefs"]
        else:
            jerk_rew = 0.0

        reward = ttc_rew + speed_rew + eco_rew + jerk_rew + tgap_rew + too_slow

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
