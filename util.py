# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
from collections import Counter, defaultdict
from itertools import takewhile, repeat
import multiprocessing
import pdb

def count_line(data_file):
	f = open(data_file, 'rb')
	bufgen = takewhile(lambda x: x, (f.read(1024*1024) for _ in repeat(None)))
	line_num = sum( buf.count(b'\n') for buf in bufgen )
	f.close()
	return line_num

def count_step(data_dir, batch_size):
	data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
	pool_size = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=pool_size)
	pool_outputs = pool.map(count_line, data_files)
	pool.close()
	pool.join()
	return sum(pool_outputs) // batch_size


