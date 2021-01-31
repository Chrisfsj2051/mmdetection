# we tested under slurm environment with Tesla V100 GPU.
# please replace 'pat_mars1' with the available partition name of your environment
bash tools/slurm_test.sh pat_mars1 train_task configs/tianchi/my_config.py mmdet_output/results
