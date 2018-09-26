kill -9 $(ps -ef|grep -E 'fengchaobing_python'|grep -v grep|awk '{print $2}')
