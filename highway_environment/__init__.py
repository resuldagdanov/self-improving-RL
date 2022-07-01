import sys
import os

from gym.envs.registration import register

# one previous directory reach
parent_directory = os.path.abspath('..')

# add parent directory to path as env forlder is located in one upper directory
sys.path.insert(0, parent_directory)

# register custom highway environment
register(
    id='highway-environment-v0',
    entry_point='highway_environment.src:Environment'
)
