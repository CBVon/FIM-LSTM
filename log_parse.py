#/usr/env/bin python
#-*- encoding: utf-8 -*- 
#Author yannnli, Mail wangzehui@sogou-inc.com

import sys,re

#EP_PATTERN = re.compile("^exp (\d+)$")
EP_PATTERN = re.compile('^epoch (\d+) starts$')
HIT_PATTERN = re.compile("top5_acc is : (.*?), top3_acc is : (.*?), top1_acc is : (.*?), top5_hit_count is : (.*?)00000, top3_hit_count is : (.*?)00000, top1_hit_count is : (.*?)00000, total count is : (.*?)00000.$")

line_print = ""
for line in sys.stdin:
	ep_res = EP_PATTERN.search(line)
	hit_res = HIT_PATTERN.search(line)
	if ep_res:
		if line_print != "":
			print(line_print)
		line_print = ep_res.group(1) + "\t"
	if hit_res:
		line_print += "\t".join(hit_res.groups())
if line_print!="":
	print line_print
