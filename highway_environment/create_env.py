import sys
import os
import gym

from gym.envs.registration import register


if __name__ == "__main__":

    # one previous directory reach
    parent_directory = os.path.abspath('..')

    # add parent directory to path
    sys.path.append(parent_directory)

    register(
        id='highway-environment-v0',
        entry_point='highway_environment.envs.environment:Environment',
    )

    env = gym.make("highway_environment:highway-environment-v0")

    print("\n Environment is successfully created as : ", env)