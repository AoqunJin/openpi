import gymnasium
import cv2
from metaworld.policies import *
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
# from gymnasium.vector.async_vector_env import AsyncVectorEnv


class MetaworldEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array', 'depth_array']}

    def __init__(self, task_name: str):
        self.env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name + "-goal-observable"](seed=42)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        self.corner_renderer = MujocoRenderer(
            self.env.model, self.env.data, None, 
            256, 256, 1000, None, 'corner'
        )
        self.gripper_renderer = MujocoRenderer(
            self.env.model, self.env.data, None, 
            256, 256, 1000, None, 'behindGripper'
        )
        self.render_mode = 'rgb_array'
        self.language_instruction = " ".join(task_name.split("-")[:-1])
        self.policy = POLICIES[task_name]()

    def _get_rgb(self):
        # NOTE: I'm using more than one camera in actual implementation
        return {
            'corner': cv2.rotate(self.corner_renderer.render('rgb_array'), cv2.ROTATE_180),
            'gripper': self.gripper_renderer.render('rgb_array'),
        }

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        state, info = self.env.reset()
        obs_rgb = self._get_rgb()
        obs = {
            "image": obs_rgb["corner"],
            "wrist_image": obs_rgb["gripper"],
            "state": state[:4]
        }
        info["full_state"] = state
        return obs, info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        obs_rgb = self._get_rgb()
        obs = {
            "image": obs_rgb["corner"],
            "wrist_image": obs_rgb["gripper"],
            "state": state[:4]
        }
        done = info["success"] or done
        info["full_state"] = state
        return obs, reward, done, truncated, info
    
    def __del__(self):
        self.env.close()


POLICIES = dict({
    "assembly-v2": SawyerAssemblyV2Policy,
    "basketball-v2": SawyerBasketballV2Policy,
    "bin-picking-v2": SawyerBinPickingV2Policy,
    "box-close-v2": SawyerBoxCloseV2Policy,
    "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy,
    "button-press-topdown-wall-v2": SawyerButtonPressTopdownWallV2Policy,
    "button-press-v2": SawyerButtonPressV2Policy,
    "button-press-wall-v2": SawyerButtonPressWallV2Policy,
    "coffee-button-v2": SawyerCoffeeButtonV2Policy,
    "coffee-pull-v2": SawyerCoffeePullV2Policy,
    "coffee-push-v2": SawyerCoffeePushV2Policy,
    "dial-turn-v2": SawyerDialTurnV2Policy,
    "disassemble-v2": SawyerDisassembleV2Policy,
    "door-close-v2": SawyerDoorCloseV2Policy,
    "door-lock-v2": SawyerDoorLockV2Policy,
    "door-open-v2": SawyerDoorOpenV2Policy,
    "door-unlock-v2": SawyerDoorUnlockV2Policy,
    "drawer-close-v2": SawyerDrawerCloseV2Policy,
    "drawer-open-v2": SawyerDrawerOpenV2Policy,
    "faucet-close-v2": SawyerFaucetCloseV2Policy,
    "faucet-open-v2": SawyerFaucetOpenV2Policy,
    "hammer-v2": SawyerHammerV2Policy,
    "hand-insert-v2": SawyerHandInsertV2Policy,
    "handle-press-side-v2": SawyerHandlePressSideV2Policy,
    "handle-press-v2": SawyerHandlePressV2Policy,
    "handle-pull-v2": SawyerHandlePullV2Policy,
    "handle-pull-side-v2": SawyerHandlePullSideV2Policy,
    "peg-insert-side-v2": SawyerPegInsertionSideV2Policy,
    "lever-pull-v2": SawyerLeverPullV2Policy,
    "peg-unplug-side-v2": SawyerPegUnplugSideV2Policy,
    "pick-out-of-hole-v2": SawyerPickOutOfHoleV2Policy,
    "pick-place-v2": SawyerPickPlaceV2Policy,
    "pick-place-wall-v2": SawyerPickPlaceWallV2Policy,
    "plate-slide-back-side-v2": SawyerPlateSlideBackSideV2Policy,
    "plate-slide-back-v2": SawyerPlateSlideBackV2Policy,
    "plate-slide-side-v2": SawyerPlateSlideSideV2Policy,
    "plate-slide-v2": SawyerPlateSlideV2Policy,
    "reach-v2": SawyerReachV2Policy,
    "reach-wall-v2": SawyerReachWallV2Policy,
    "push-back-v2": SawyerPushBackV2Policy,
    "push-v2": SawyerPushV2Policy,
    "push-wall-v2": SawyerPushWallV2Policy,
    "shelf-place-v2": SawyerShelfPlaceV2Policy,
    "soccer-v2": SawyerSoccerV2Policy,
    "stick-pull-v2": SawyerStickPullV2Policy,
    "stick-push-v2": SawyerStickPushV2Policy,
    "sweep-into-v2": SawyerSweepIntoV2Policy,
    "sweep-v2": SawyerSweepV2Policy,
    "window-close-v2": SawyerWindowCloseV2Policy,
    "window-open-v2": SawyerWindowOpenV2Policy,
})
