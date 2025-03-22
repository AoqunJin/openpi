"""
Usage:
python examples/metaworld/convert_metaworld_to_lerobot.py --input_path /path/to/your/data --output_path /path/to/your/data

"""

import shutil
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro


def main(input_path: str, output_path: str, *, push_to_hub: bool = False):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=str(output_path),
        robot_type="sawyer",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (4,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (4,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for npz_file in input_path.glob("**/*.npz"):
        data = np.load(npz_file)
        for step in range(len(data["images"])):
            dataset.add_frame({
                "image": data["images"][step],
                "wrist_image": data["wrist_images"][step],
                "state": data["states"][step],
                "actions": data["actions"][step],
            })
        dataset.save_episode(task=data["language_instruction"].item())
    
    dataset.consolidate(run_compute_stats=False)
    
    if push_to_hub:
        dataset.push_to_hub(
            tags=["metaworld", "sawyer", "lerobot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
