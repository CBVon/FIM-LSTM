#!/bin/bash

#rm -rf ./model/*
#rm -rf ./log/*

log_file=./log/dynamic_share_lstm.log

echo "exp starts">$log_file

CUDA_VISIBLE_DEVICES='' ./fengchaobing_python -u train.py --ps_hosts=10.141.104.68:2332 --worker_hosts=10.141.104.68:2333,10.141.104.68:2334,10.141.104.68:2335,10.141.104.68:2336 --job_name=ps --task_index=0 1>>${log_file} 2>&1 &
CUDA_VISIBLE_DEVICES='0' ./fengchaobing_python -u train.py --ps_hosts=10.141.104.68:2332 --worker_hosts=10.141.104.68:2333,10.141.104.68:2334,10.141.104.68:2335,10.141.104.68:2336 --job_name=worker --task_index=0 1>>${log_file} 2>&1 &
CUDA_VISIBLE_DEVICES='1' ./fengchaobing_python -u train.py --ps_hosts=10.141.104.68:2332 --worker_hosts=10.141.104.68:2333,10.141.104.68:2334,10.141.104.68:2335,10.141.104.68:2336 --job_name=worker --task_index=1 1>>${log_file} 2>&1 &
CUDA_VISIBLE_DEVICES='2' ./fengchaobing_python -u train.py --ps_hosts=10.141.104.68:2332 --worker_hosts=10.141.104.68:2333,10.141.104.68:2334,10.141.104.68:2335,10.141.104.68:2336 --job_name=worker --task_index=2 1>>${log_file} 2>&1 &
CUDA_VISIBLE_DEVICES='3' ./fengchaobing_python -u train.py --ps_hosts=10.141.104.68:2332 --worker_hosts=10.141.104.68:2333,10.141.104.68:2334,10.141.104.68:2335,10.141.104.68:2336 --job_name=worker --task_index=3 1>>${log_file} 2>&1 &

