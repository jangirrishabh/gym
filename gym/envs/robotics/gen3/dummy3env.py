import os
from gym import utils
from gym.envs.robotics import dummy_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('gen3', 'cloth_corner.xml')


class Dummy3Env(dummy_env.DummyEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {

        }
        dummy_env.DummyEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, has_cloth=False, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
