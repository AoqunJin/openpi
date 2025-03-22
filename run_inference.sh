export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# Run server
python scripts/serve_policy.py \
    --env METAWORLD \
    policy:checkpoint \
    --policy.config pi0_metaworld_low_mem \
    --policy.dir /home/jinaoqun/workspace/openpi/.cache/models/checkpoints_mt10/pi0_metaworld_low_mem_finetune/metaworld_mt10/29999


# Run client
# python examples/metaworld/main.py --args.num_trials_per_task 10
