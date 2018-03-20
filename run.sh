#!/bin/bash

gpu=0
save_path='save/baseline/Market/rec1_cls1_isTransfer'
weight_L1=1
weight_softmax=1
target='Market'
source='Duke'
print_freq=50
plot_freq=10
python main.py --save_path ${save_path} --weight_L1 ${weight_L1} --weight_softmax ${weight_softmax} --gpu ${gpu} --target ${target} --source ${source} --print_freq ${print_freq} --plot_freq ${plot_freq} --is_transfer_net