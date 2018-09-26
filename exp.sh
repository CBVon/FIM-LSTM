#!/bin/bash

log_file='./exp.log'
exp_file='../../data/testsets/up_total_small_transfered_digit_nohead'
echo "exp_starts" > $log_file
epoches=(322002 276003 138003 92004)
for epoch in ${epoches[*]}
do
	sed "s/111602/$epoch/g" predict.tmp.py > predict.test.py
	./fengchaobing_python ./predict.test.py $exp_file > ./exp.output
	echo "epoch "$epoch" starts" >> $log_file
	tail -1 exp.output >> $log_file
done
