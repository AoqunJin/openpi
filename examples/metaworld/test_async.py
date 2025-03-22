import multiprocessing as mp 
import metaworld
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.vector.async_vector_env import AsyncVectorEnv


class MetaworldEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array', 'depth_array']}

    def __init__(self):
        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name](seed=seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.corner_renderer = MujocoRenderer(
            self.env.model, self.env.data, None, 
            128, 128, 1000, None, 'corner', {}
        )
        self.render_mode = 'rgb_array'

    def _get_rgb(self):
        # NOTE: I'm using more than one camera in actual implementation,
        # here is just for error reproduction, so I'm using only one camera
        return {
            'default': self.env.mujoco_renderer.render('rgb_array'),    # <-- This is the default metaworld
            'corner': self.corner_renderer.render('rgb_array'),         #     renderer, and it also cannot
        }                                                               #     work. Mine too.

    def reset(self, **kwargs):
        self.env.reset()
        self.env.reset_model()
        state = self.env.reset()
        obs_dict = self._get_rgb()      # <-- This is the line that causes the error
        obs_dict['full_state'] = state
        return obs_dict

def env_fn():
    return MetaworldEnv()


if __name__ == "__main__":
    task_name = 'shelf-place-v2-goal-observable'
    seed = 42
    n_env = 2
    env = AsyncVectorEnv([env_fn for _ in range(n_env)])
    env.reset()
    