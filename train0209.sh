conda activate avenger
nohup python train.py --total_epoch 100 --check_period 10 --output_dir './output/output_0209' >output/train0209.out 2>output/train0209.err &
