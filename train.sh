# we trained under slurm environment with 2 nodes, 16 Tesla V100 GPU in total (8/node).
# please replace 'pat_mars1' with the available partition name of your environment
# please make sure that your checkpoint path is ../user_data/model_data/ckpt.pth
GPUS=16 bash tools/slurm_train.sh pat_mars1 test_task configs/tianchi/my_config.py ../user_data/model_data/ckpt.pth --format-only
mv submit.json ../prediction_result/result.json
