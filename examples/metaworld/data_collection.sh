# export MUJOCO_GL=egl 
export MUJOCO_GL=osmesa


# Collect data
python examples/metaworld/collect_metaworld_to_lerobot.py \
    --save_path /home/jinaoqun/workspace/openpi/.cache/data/metaworld_mt50 \
    --benchmark mt50

# Data to lerobot
python examples/metaworld/convert_metaworld_to_lerobot.py \
    --input_path /home/jinaoqun/workspace/openpi/.cache/data/metaworld_mt50 \
    --output_path /home/jinaoqun/workspace/openpi/.cache/lerobot_dataset/metaworld_mt50
