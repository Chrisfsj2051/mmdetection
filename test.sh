# we tested under slurm environment with Tesla V100 GPU.
# please replace 'pat_mars1' with the available partition name of your environment
# please make sure that your checkpoint path is ../user_data/model_data/ckpt.pth
bash tools/slurm_test.sh pat_mars1 test_task configs/tile/my_config.py ../user_data/model_data/ckpt.pth --format-only

# if the slurm is not available, you may use the following command, though we haven't testify it.
# bash tools/dist_test.sh configs/tile/my_config.py ../user_data/model_data/ckpt.pth 8 --format-only

mv submit.json ../prediction_result/result.json
