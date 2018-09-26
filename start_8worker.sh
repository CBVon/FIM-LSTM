#!/bin/bash
echo "exp starts">./log/dynamic_share_lstm.log
#rm -rf ./model/*
#rm -rf ./log/*

CUDA_VISIBLE_DEVICES='' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238 --job_name=ps --task_index=0 1>>./log/dynamic_share_lstm.log 2>&1 &
CUDA_VISIBLE_DEVICES='0' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238 --job_name=worker --task_index=0 1>>./log/dynamic_share_lstm.log 2>&1 &
CUDA_VISIBLE_DEVICES='1' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238 --job_name=worker --task_index=1 1>>./log/dynamic_share_lstm.log 2>&1 &
CUDA_VISIBLE_DEVICES='2' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238 --job_name=worker --task_index=2 1>>./log/dynamic_share_lstm.log 2>&1 &
CUDA_VISIBLE_DEVICES='3' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238 --job_name=worker --task_index=3 1>>./log/dynamic_share_lstm.log 2>&1 &
CUDA_VISIBLE_DEVICES='4' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238 --job_name=worker --task_index=4 1>>./log/dynamic_share_lstm.log 2>&1 &
CUDA_VISIBLE_DEVICES='5' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238 --job_name=worker --task_index=5 1>>./log/dynamic_share_lstm.log 2>&1 &
CUDA_VISIBLE_DEVICES='6' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238  --job_name=worker --task_index=6 1>>./log/dynamic_share_lstm.log 2>&1 &
CUDA_VISIBLE_DEVICES='7' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2230 --worker_hosts=10.141.160.45:2231,10.141.160.45:2232,10.141.160.45:2233,10.141.160.45:2234,10.141.160.45:2235,10.141.160.45:2236,10.141.160.45:2237,10.141.160.45:2238 --job_name=worker --task_index=7 1>>./log/dynamic_share_lstm.log 2>&1 &

#CUDA_VISIBLE_DEVICES='' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2222 --worker_hosts=10.141.160.45:2223 --job_name=ps --task_index=0 &
##CUDA_VISIBLE_DEVICES='1' ./zahui_python -u train.py --ps_hosts=10.141.160.45:2222 --worker_hosts=10.141.160.45:2223 --job_name=worker --task_index=0






#CUDA_VISIBLE_DEVICES='' ./zahui_python predict.py
