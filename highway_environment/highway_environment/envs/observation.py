import gym
import numpy as np
import pandas as pd

from typing import List, Dict

from highway_env import utils
from highway_env.envs.common.observation import ObservationType


class Observation(ObservationType):

    FEATURES: List[str] = [
        "ego_speed",
        "ego_accel",
        "ego_jerk",
        "mio_pos",
        "mio_vel"
    ]
    FEATURES_RANGE: Dict[str, List[float]] = {
        "ego_speed":    [  0.0,  45.0 ], # [m/s]
        "ego_accel":    [ -4.0,   4.0 ], # [m/s^2]
        "ego_jerk":     [ -8.0,   8.0 ], # [m/s^2]
        "mio_pos":      [  0.0, 150.0 ], # [m]
        "mio_vel":      [-40.0,  40.0 ], # [m/s]
    }

    def __init__(self, env: 'AbstractEnv', frequency: int,
                features: List[str] = None, features_range: Dict[str, List[float]] = None,
                normalize: bool = True, clip: bool = True, **kwargs: dict) -> None:
        
        super().__init__(env)

        self.features = features or self.FEATURES
        self.features_range = features_range or self.FEATURES_RANGE

        self.normalize = normalize
        self.clip = clip
        self.frequency = frequency

        self.initial = True
        self.prev_speed = 0.0
        self.prev_accel = 0.0

    def space(self) -> gym.spaces.Space:
        return gym.spaces.Box(shape=(len(self.features), ), low=-1.0, high=1.0, dtype=np.float32)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        
        return df

    def observe(self) -> np.ndarray:
        # NOTE: designed only for single lane scenario.
        # for multi lane scenario increase count in "close_vehicles_to" then check lanes of vehicles to find mio!
        if not self.env.road:
            return np.zeros(self.space().shape)[0]

        ego_speed = self.observer_vehicle.to_dict()["vx"]

        # df = pd.DataFrame.from_records({"ego_speed": ego_speed}, index=[0])
        
        if self.initial:
            accel = 0.0
            jerk = 0.0
            self.initial = False
        
        else:
            accel = (ego_speed - self.prev_speed) * self.frequency
            jerk = (accel - self.prev_accel) * self.frequency
        
        self.prev_speed = ego_speed
        self.prev_accel = accel

        # add nearby traffic
        vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=1,
            see_behind=False,
            sort=True
        )

        mio = vehicles[0].to_dict(self.observer_vehicle, observe_intentions=False) if len(vehicles) > 0 else {"presence": False}
        
        if mio["presence"] == 1:
            mio_pos = mio["x"]
            mio_vel = mio["vx"]
            
        else:
            mio_pos = self.features_range["mio_pos"][1]
            mio_vel = self.features_range["mio_vel"][1]
        
        # global position and velocity of mio
        mio_location = mio_pos + self.observer_vehicle.to_dict()["x"]
        mio_speed = mio_vel + ego_speed
        
        # calculate time-gap and time-to-collision
        tgap = mio_pos / (ego_speed + 1e-5) if (ego_speed + 1e-5) != 0.0 else 9999
        ttc = mio_pos / (mio_vel - 1e-5) if (mio_vel - 1e-5) != 0.0 else 9999

        self.raw_obs = {
            "ego_speed": ego_speed,
            "ego_accel": accel,
            "ego_jerk": jerk,
            "mio_pos": mio_pos,
            "mio_vel": mio_vel,
            "tgap": tgap,
            "ttc": ttc
        }
        
        df = pd.DataFrame(self.raw_obs, index=[0])
        
        # normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        
        # reorder
        df = df[self.features]
        obs = df.values.copy()
        
        # flatten
        return obs.astype(self.space().dtype)[0]
