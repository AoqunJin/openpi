"""
Usage:
MUJOCO_GL=egl python examples/metaworld/collect_metaworld_to_lerobot.py --save_path /path/to/your/data --benchmark mt10

"""

import shutil
import numpy as np
import metaworld.envs.mujoco.env_dict as env_dict
import tyro
from pathlib import Path
from metaworld_tools import MetaworldEnv

RAW_DATASET_NAME_DICT = {
    "mt10": env_dict.MT10_V2.keys(),
    "mt50": env_dict.MT50_V2.keys(),
    "ml10": env_dict.ML10_V2["train"].keys(),
    "ml45": env_dict.ML45_V2["train"].keys(),
}

def main(benchmark: str, save_path: str):
    output_path = Path(save_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Loop over MetaWorld tasks and collect data
    for env_name in RAW_DATASET_NAME_DICT[benchmark]:
        env = MetaworldEnv(env_name)
        task_path = output_path / env_name
        task_path.mkdir(parents=True, exist_ok=True)
        
        for episode_idx in range(50):  # Modify as needed
            obs, info = env.reset()
            done = False
            truncated = False
            episode_data = {
                "images": [], "wrist_images": [], "states": [], "actions": [],
            }
            
            while not (truncated or done):
                action = env.policy.get_action(info["full_state"])
                next_obs, reward, done, truncated, info = env.step(action)
                
                episode_data["images"].append(obs["image"])
                episode_data["wrist_images"].append(obs["wrist_image"])
                episode_data["states"].append(obs["state"])
                episode_data["actions"].append(action)
                
                obs = next_obs
            
            np.savez_compressed(
                task_path / f"episode_{episode_idx}.npz",
                images=np.array(episode_data["images"], dtype=np.uint8),
                wrist_images=np.array(episode_data["wrist_images"], dtype=np.uint8),
                states=np.array(episode_data["states"], dtype=np.float32),
                actions=np.array(episode_data["actions"], dtype=np.float32),
                language_instruction=env.language_instruction
            )

if __name__ == "__main__":
    tyro.cli(main)
