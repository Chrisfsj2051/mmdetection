# we trained under slurm environment with 2 nodes, 16 Tesla V100 GPU in total (8/node).
# please replace 'pat_mars1' with the available partition name of your environment
GPUS=16 bash tools/slurm_train.sh pat_mars1 train_task configs/tile/my_config.py mmdet_output/results
