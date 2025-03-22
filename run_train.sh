export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# python scripts/compute_norm_stats.py --config-name=pi0_metaworld_low_mem_finetune
# mv ./assets/* .cache/assets/

python scripts/train.py pi0_metaworld_low_mem_finetune \
    --exp-name metaworld_mt10 \
    --weight-loader.params-path /home/jinaoqun/workspace/openpi/.cache/models/pi0_base/params \
    --checkpoint-base-dir /home/jinaoqun/workspace/openpi/.cache/models/checkpoints_mt10 \
    --num-train-steps 30000 \
    --save-interval 10000 \
    --overwrite \
    --data.repo-id lerobot_dataset/metaworld_mt10 \
    --assets-base-dir .cache/assets \
    --data.assets.asset-id lerobot_dataset/metaworld \
    --batch-size 4 \
    --no-wandb-enabled
