#!/usr/env/bin python
#-*- coding:utf-8 -*-
#This script is used to output the steps of model, and you should input the model_dir
#Author yannnli, Mail wangzehui@sogou-inc.com

import sys,os,re

EPOCH_PATTERN = re.compile('(.*?)[-_]+(\d+).meta$')

if __name__ == "__main__":
	if len(sys.argv[1])<2:
		sys.stderr.write('You should use the script as : python model_step_get.py model_dir\n')
	else:
		model_dir = sys.argv[1]
		if not os.path.isdir(model_dir):
			sys.stderr.write('ERROR MODEL DIR : '+model_dir+'\n')
		else:
			filenames = os.listdir(model_dir)
			steps = []
			for filename in filenames:
				res = EPOCH_PATTERN.search(filename)
				if res:
					steps.append(int(res.group(2)))
	steps.sort()
	for step in steps[::-1]:
		sys.stdout.write(str(step)+' ')
