# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import pdb
import random
import collections
import threading
import math
import sys
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from ortools.graph import pywrapgraph
from multiprocessing import Process, Queue, Pool
#from dict_adjuster import dict_adjuster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
flags = tf.app.flags
logging = tf.logging

flags.DEFINE_string(
		"model", "small",
		"A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "../../data/", "data_path")
flags.DEFINE_string("dataset", "ugc-batch5", "The dataset we use for training")
flags.DEFINE_string("model_dir", "./model/", "model_path")

# Flags for defining the tf.train.ClusterSpec
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_string('RNNCell', "LSTM", 'Cell name in RNN')
flags.DEFINE_integer('num_layers', 1, 'Number of layers in RNN')
flags.DEFINE_integer('num_steps', 20, 'Number of steps for BPTT')
flags.DEFINE_integer('hidden_size', 512, 'Number of hidden nodes for one layer')
flags.DEFINE_integer('embedding_size', 256, 'embedding size')
flags.DEFINE_integer('thread_num', 3, 'num of thread per queue')
flags.DEFINE_integer('max_epoch', 50, 'Number of epochs before stop')
flags.DEFINE_integer('batch_size', 320, 'Number of lines in one batch for training')
flags.DEFINE_integer('vocab_size', 8100, 'Size of vocabulary')
flags.DEFINE_integer('top_num', 30, 'num of top candidates when calculate accuracy')
flags.DEFINE_float("lr_decay_factor", 0.5, "The decay factor for learning rate")
flags.DEFINE_float("initial_lr", 0.1, "The initial learning rate for training model")
flags.DEFINE_float("lstm_keep_prob", 0.5, "The keep rate for lstm layers")
flags.DEFINE_float("input_keep_prob", 0.8, "The keep rate for input layer")
flags.DEFINE_float("max_grad_norm", 1.0, "The max norm that clip the gradients")
flags.DEFINE_float("converge_rate", 0.01, "The converge rate the we tell the training is converged")
flags.DEFINE_bool("use_adam", False, "Use AdamOptimizer as training optimizer")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")

flags.DEFINE_bool("save", False, "Save model")
flags.DEFINE_bool("restore", True, "Restore model")
flags.DEFINE_bool("predict", True, "Prediction topK")
flags.DEFINE_bool("train", False, "TrainModel Flag")
FLAGS = flags.FLAGS
train_path = os.path.join(FLAGS.data_path, FLAGS.dataset, "%s.train.resort.txt" % FLAGS.dataset)
valid_path = os.path.join(FLAGS.data_path, FLAGS.dataset, "%s.valid.resort.txt" % FLAGS.dataset)
test1_path = os.path.join(FLAGS.data_path, FLAGS.dataset, "%s.test.resort.txt" % FLAGS.dataset)
test2_path = os.path.join(FLAGS.data_path, FLAGS.dataset, "%s.test1.txt" % FLAGS.dataset)
long_path = os.path.join(FLAGS.data_path, FLAGS.dataset, "long_all")
dic_name = FLAGS.dataset+".dic"
model_name = FLAGS.dataset+"_"+FLAGS.RNNCell+".model"
test_name_log = FLAGS.dataset+"_"+FLAGS.RNNCell+".testlog.txt"
vocab_path = os.path.join(FLAGS.data_path, FLAGS.dataset, dic_name)
model_dir = os.path.join(FLAGS.model_dir, FLAGS.dataset, FLAGS.RNNCell)
checkpoint_path = os.path.join(model_dir, model_name)


class Option(object):
	def __init__(self, mode):
		self.mode = mode
		self.num_layers = FLAGS.num_layers
		self.embedding_size = FLAGS.embedding_size	
		self.hidden_size = FLAGS.hidden_size
		self.RNNCell = FLAGS.RNNCell
		self.vocab_size = FLAGS.vocab_size
		self.top_num = FLAGS.top_num
		self.initial_lr = FLAGS.initial_lr
		self.max_grad_norm = FLAGS.max_grad_norm
		self.use_adam = FLAGS.use_adam
		self.batch_size = FLAGS.batch_size if self.mode != "predict" else 1
		self.num_steps = FLAGS.num_steps
		self.lstm_keep_prob = FLAGS.lstm_keep_prob if self.mode == "train" else 1.0
		self.input_keep_prob = FLAGS.input_keep_prob if self.mode == "train" else 1.0
def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32

def ptb_raw_data():
	if os.path.isfile(vocab_path):
		print (vocab_path)
		word2id_file = open(vocab_path, 'rb')
		words = pickle.load(word2id_file)
		if words.count("<eos>")==0:#Modified by yannnli, used to make sure the <eos> is of lable 0
			words.insert(0,"<eos>")
		eos_index = words.index("<eos>")
		if eos_index != 0:
			first_ele = words[0]
			words[0] = "<eos>"
			words[eos_index] = first_ele
		if "<bos>" not in words:
			words.insert(1, "<bos>")#End of yannnli modification
		word_dic = words[:FLAGS.vocab_size]
		word_to_id = dict(zip(word_dic, range(len(word_dic))))
		word2id_file.close()
	else:
		word_to_id, word_dic = build_vocab()			#gone': 17, 'bert': 9, 'bris': 10, 
	dicfile = open("dic.txt", "w")
	for i in range(len(word_dic)):
		dicfile.write(word_dic[i])
		dicfile.write("\n")
	dicfile.close()
	vocabulary_len = len(word_to_id)
	print("vocabulary size:%d"%(vocabulary_len))
	train_data, train_data_len = file_to_word_ids(train_path, word_to_id)  # list
	valid_data, valid_data_len = file_to_word_ids(valid_path, word_to_id)
	test1_data, test1_data_len = file_to_word_ids(test1_path, word_to_id)
	test2_data, test2_data_len = file_to_word_ids(test2_path, word_to_id)
	print ("train_line:%d"% len(train_data))
	train_step = len(train_data) // FLAGS.batch_size 
	print ("Valid_line:%d"% len(valid_data))
	valid_step = len(valid_data) // FLAGS.batch_size
	print ("test1_line:%d"% len(test1_data))
	test1_step = len(test1_data) // FLAGS.batch_size
	print ("test2_line:%d"% len(test2_data))
	test2_step = len(test2_data) // FLAGS.batch_size
	pad_id = word_to_id["<eos>"]
	return train_data, train_data_len, valid_data, valid_data_len, test1_data, test1_data_len, test2_data, test2_data_len, vocabulary_len, word_to_id, word_dic, train_step, valid_step, test1_step, test2_step, pad_id

def file_to_word_ids(filename, word_to_id):
	#data = read_words(filename)
	f=open(filename)
	seq_list=[]
	seq_len_list=[]
	line_num = 0
	less_numsteps_line_num = 0
	for line in f:
		line_num += 1
		count=["<bos>"]
		count.extend(line.replace("\n"," <eos> ").split())
		len_count = len(count)
		count_id = [word_to_id[word] if word_to_id.has_key(word) else word_to_id["<unk>"] for word in count]
		seq_len_list.append(len_count-1)
		seq_list.append(count_id)
	#seq_list.sort(key=len)
	batch_num = len(seq_list) // FLAGS.batch_size
	pad_id = word_to_id["<eos>"]
	for i in range(batch_num):
		#batch_data = seq_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
		max_len = len(max(seq_list[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], key=len))
		for j in range(FLAGS.batch_size):
			for k in range(max_len - seq_len_list[i*FLAGS.batch_size+j]):
				seq_list[i*FLAGS.batch_size+j].append(pad_id)
	assert(line_num == len(seq_list))
	print ("line_num:%d"%line_num)
	#print ("less_numsteps_line_num:%d"%less_numsteps_line_num)
	return seq_list, seq_len_list#[word_to_id[word] if word_to_id.get(word) else word_to_id["<unk>"] for word in data]

def read_words(filename):  # return 1-D list
	f=open(filename)
	content=[]
	for line in f:
		temp = ["<bos>"]
		temp.extend(line.replace("\n"," <eos> ").split())
		content.extend(temp)
	f.close()
	return content

def build_vocab():
	data = read_words(train_path)
	data.extend(read_words(valid_path))
	data.extend(read_words(test1_path))
	counter = collections.Counter(data)  # sort words '.': 5, ',': 4......
	#counter_items = [i for i in counter.items() if i[1] >= 10]
	#count_pairs = sorted(counter_items, key=lambda x: (-x[1], x[0]))[:FLAGS.vocab_size-1]	# make it pair list, ('.', 5)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))	# make it pair list, ('.', 5)
	words = list(zip(*count_pairs)[0])
	if '<unk>' not in words:
		words.insert(0, "<unk>")
	print ("origin voc len:%d"%len(words))
	words = words[:FLAGS.vocab_size]
	dic_file = open(vocab_path, 'wb')
	pickle.dump(words, dic_file)
	dic_file.close()
	word_to_id = dict(zip(words, range(len(words))))	 #š'gone': 17, 'bert': 9, 'bris': 10, 
	return word_to_id, words

class LockedGen(object):
	def __init__(self, it):
		self.lock = threading.Lock()
		self.it = it.__iter__()

	def __iter__(self): 
		return self

	def next(self):
		self.lock.acquire()
		try:
			return self.it.next()
		finally:
			self.lock.release()
def start_threads_func(sess, coord):
	def feed_queue_data(data_gen, model, sess, coord):
		while not coord.should_stop():
			try:
				data_x, data_y, data_x_len = data_gen.next()
				#print ("data_x, data_y")
				sess.run(model.enqueue_op, feed_dict={model.x:data_x, model.y:data_y, model.x_len:data_x_len})
				#print ("queue size:%d"%(sess.run(model.queue_size)))
			except StopIteration:
				# Data finished for one epoch
				coord.request_stop()
				break
	
	def start_threads(data, data_len, model, thread_num, pad_id):
		# Create fresh generator every time you call start_threads
		data_gen = LockedGen(get_next_batch(data, data_len, pad_id))
		threads = []
		for i in range(thread_num):
			t = threading.Thread(target=feed_queue_data, args=(data_gen, model, sess, coord))
			threads.append(t)
		for thread in threads:
			thread.daemon = True
			thread.start()
		return threads

	return start_threads
def get_next_batch(input_data, input_data_len, pad_id):
	batch_num = len(input_data) // FLAGS.batch_size
	#print ("epoch_size:%d"%epoch_size)
	if batch_num == 0:
		raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
	for i in range(batch_num):
		x = input_data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
		#print("x len :%d"%len(x))
		y = input_data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
		for j in range(FLAGS.batch_size):
			x[j] = x[j][:-1]
			y[j] = y[j][1:]
		seq_len = input_data_len[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
		yield (x, y, seq_len)

def ptb_iterator(raw_data, deviceID, workernum):
	batch_size = FLAGS.batch_size
	num_steps = FLAGS.num_steps
	raw_data = np.array(raw_data, dtype=np.int32)
	device_data_len = len(raw_data) // workernum
	batch_len = device_data_len // batch_size 
	data = np.zeros([batch_size, batch_len], dtype=np.int32)
	for i in range(batch_size):
		data[i] = raw_data[deviceID * device_data_len + batch_len * i : deviceID * device_data_len + batch_len * (i + 1)]
	epoch_size = (batch_len - 1) // num_steps
	print ("epoch_size:%d"%epoch_size)
	if epoch_size == 0:
		raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
	for i in range(epoch_size):
		x = data[:, i*num_steps:(i+1)*num_steps]
		y = data[:, i*num_steps+1:(i+1)*num_steps+1]
		yield (x, y)

def ptb_iterator_pred(raw_data):
	num_steps = FLAGS.num_steps
	raw_data = np.array(raw_data, dtype=np.int32)
	data_len = len(raw_data)
	x = np.zeros([1, num_steps], dtype=np.int32)
	if num_steps < data_len:
		x[0] = raw_data[-num_steps:]
	else:
		x[0][-data_len:] = raw_data
	yield (x) 

class RNN_Model(object):
	
	def __init__(self, opt, reuse=False):
		
		with tf.variable_scope("{}_input".format(opt.mode)):
			#ddwith tf.device("/job:ps/task:0"):
			if True:
				self.x = tf.placeholder(dtype=tf.int32, shape=[opt.batch_size, None], name="x")
				self.x_len = tf.placeholder(dtype=tf.int32, shape=[opt.batch_size], name="x_len")
				print("Load successfully...")
		#pdb.set_trace()
		with tf.variable_scope("model", reuse=reuse), tf.name_scope("{}_model".format(opt.mode)):
			self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
			#self.input_data,self.seq_len = queue_outputs
			self.input_data = self.x
			self.seq_len = self.x_len
			stdv = np.sqrt(1. / opt.vocab_size)
			embedding = tf.get_variable("embedding_r", [opt.vocab_size, opt.embedding_size], initializer=tf.random_uniform_initializer(-stdv, stdv))
			inputs = tf.nn.embedding_lookup(embedding, self.input_data)
			inputs = tf.nn.dropout(inputs, opt.input_keep_prob) 
			if opt.RNNCell == "LSTM":
				lstm_cell = tf.contrib.rnn.BasicLSTMCell(opt.hidden_size, forget_bias=1.0, state_is_tuple=True) 
			elif opt.RNNCell == "GRU":
				lstm_cell = tf.contrib.rnn.GRUCell(opt.hidden_size)
			lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=opt.lstm_keep_prob)
			cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * opt.num_layers, state_is_tuple=True)
			self.initial_state = cell.zero_state(opt.batch_size, data_type())
			self.state = self.initial_state
		
			#pdb.set_trace()
			softmax_w = tf.get_variable("softmax_w", [opt.hidden_size, opt.vocab_size], dtype=data_type())
			softmax_b = tf.get_variable("softmax_b", [opt.vocab_size], dtype=data_type())
			
			self.cell_outputs = []
			with tf.variable_scope("RNN"):
				self.cell_outputs, self.state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.seq_len, initial_state=self.initial_state)

			# Evaluate model
			self.output = tf.reshape(tf.concat(self.cell_outputs, 1), [-1, opt.hidden_size])
			self.logits = tf.matmul(self.output, softmax_w) + softmax_b
			self.softmax_logits = tf.nn.softmax(self.logits)
			self.pred_val, self.pred_topK = tf.nn.top_k(self.logits, opt.top_num)

	def update_lr(self, session, new_lr):
		if self.is_training:
			session.run(self.lr_decay_op, feed_dict={self.new_lr: new_lr})	


def cal_dynamic_mean(f_top_k,x_len,batch_size,step_column):
	acc_sum = 0.0
	for i in range(batch_size):
		x_len_size = x_len[i]
		sum_tmp = 0.0
		for j in range(0,x_len_size-1):
			sum_tmp += f_top_k[i*step_column+j]
		acc_sum += sum_tmp/float(x_len_size-1)
	return acc_sum/float(batch_size)
	
def train():
	ps_hosts = FLAGS.ps_hosts.split(",")
	worker_hosts = FLAGS.worker_hosts.split(",")
	workernum = len(worker_hosts)
	
	# Create a cluster from the parameter server and worker hosts.
	cluster = tf.train.ClusterSpec({ "ps": ps_hosts, "worker" : worker_hosts })
	
	# start a server for a specific task
	server = tf.train.Server(cluster, 
														job_name=FLAGS.job_name,
														task_index=FLAGS.task_index)
		
	if FLAGS.job_name == "ps":
		server.join()
	elif FLAGS.job_name == "worker":
		is_chief = FLAGS.task_index == 0
		if is_chief:
			train_data, train_data_len, valid_data, valid_data_len, test1_data, test1_data_len, test2_data, test2_data_len, vocab_len , word_to_id, word_dic, train_step, valid_step, test1_step, test2_step, pad_id = ptb_raw_data()
			print ("train_step:%d, valid_step:%d, test1_step:%d, test2_step:%d"%(train_step, valid_step, test1_step, test2_step))
		
		with tf.device('/cpu:0'):
			# Define training variables and ops
			epoch_var = tf.get_variable("epoch", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
			increment_epoch = tf.assign_add(epoch_var, 1)
			epoch_var_assign_op = epoch_var.assign(0)
		

		with tf.device(tf.train.replica_device_setter(
					worker_device="/job:worker/task:%d" % FLAGS.task_index,
					cluster=cluster)): 
			print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
			train_opt = Option("train")
			valid_opt = Option("valid")
			test_opt = Option("test")

			train_model = RNN_Model(train_opt, False)
			valid_model = RNN_Model(valid_opt, True)
			test_model = RNN_Model(test_opt, True)
			

		print("Variables initialized ...")
		mysaver = tf.train.Saver(tf.global_variables())
		init_op = tf.global_variables_initializer()
		sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
																	global_step=train_model.global_step,
																	init_op=init_op,
																	saver=mysaver)
		sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=False)

		with sv.prepare_or_wait_for_session(server.target, config = sess_config) as sess:
			if not os.path.isdir(model_dir):
				os.makedirs(model_dir)
			if FLAGS.restore:
				ckpt = tf.train.get_checkpoint_state(model_dir)
				if ckpt: #and tf.gfile.Exists(ckpt.model_checkpoint_path): 
					print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
					sv.saver.restore(sess, ckpt.model_checkpoint_path)
					sess.run(epoch_var_assign_op)
				else:
					print("No checkpoint file found!")
					return
			#else:
			#	sess.run(init_op)
			if is_chief:
				coord = tf.train.Coordinator()
				start_threads = start_threads_func(sess, coord)
			ppl_history = []
			current_epoch = 0
			run_options = tf.RunOptions(timeout_in_ms=10000, trace_level=tf.RunOptions.FULL_TRACE)
			for epoch in range(FLAGS.max_epoch):
					start_time = time.time()
					print("Time:%s Worker_ID:%d"%(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), FLAGS.task_index))
					costs = 0.0
					if is_chief:
						threads = start_threads(train_data, train_data_len, train_model, FLAGS.thread_num, pad_id)
					iters = 0
					step = 0
					#pdb.set_trace()
					run_metadata = tf.RunMetadata()
					acc_topk_list=[]
					acc_top1_list=[]
					print("worker_id:%d current_epoch:%d epoch_var:%d"%(FLAGS.task_index, current_epoch, epoch_var.eval()))
					while current_epoch == epoch_var.eval():
						try:
							cost_val,_,logits,outp,cell_out= sess.run([train_model.loss, train_model.train_op,train_model.logits,train_model.output,train_model.cell_outputs], options=run_options, run_metadata=run_metadata)
							tl = timeline.Timeline(run_metadata.step_stats)
							ctf = tl.generate_chrome_trace_format()
							with open('timeline.json', 'w') as f:
								f.write(ctf)
							iters += 1
							costs += cost_val
							if step % 30 == 0:
								if step > 0:
									sv.saver.save(sess, checkpoint_path, global_step=epoch)
								print("	doing step %d, queue size:%d, PPL:%f, speed: %f , worker_id: %d" % 
											(step, sess.run(train_model.queue_size), np.exp(costs / iters), iters*FLAGS.batch_size/(time.time()-start_time), FLAGS.task_index))
							step += 1
						except tf.errors.DeadlineExceededError:
							if is_chief and coord.should_stop():
								coord.join(threads, stop_grace_period_secs=10)
								coord.clear_stop()
								sess.run(increment_epoch)
							pass
					if iters > 0:
						train_ppl= np.exp(costs / iters)
						print("TaskID:%d, Train epoch: %d  Train Perplexity: %.4f speed: %.0f wps"% (FLAGS.task_index, epoch, train_ppl, iters*FLAGS.batch_size/(time.time()-start_time)))
					current_epoch += 1

					acc_topk_list=[]
					acc_top1_list=[]
					##Valid
					start_time = time.time()
					costs = 0.0
					if is_chief:
						threads = start_threads(valid_data, valid_data_len, valid_model, FLAGS.thread_num, pad_id)
					iters = 0
					step = 0
					while current_epoch == epoch_var.eval():
						try:
							valid_input,seq_len,cost_val,f_topk,f_top1 = sess.run([valid_model.input_data,valid_model.seq_len,valid_model.loss,valid_model.f_top_k,valid_model.f_top_1], options=run_options)
							step_column = (valid_input.shape)[-1]
							acc_topk = cal_dynamic_mean(f_topk,seq_len,FLAGS.batch_size,step_column)
							acc_top1 = cal_dynamic_mean(f_top1,seq_len,FLAGS.batch_size,step_column)
							acc_topk_list.append(acc_topk)
							acc_top1_list.append(acc_top1)
							iters += 1
							costs += cost_val
							step += 1
							#if step > 0 and step % 100 == 0:
								#print("	doing step %d, queue size:%d, worker_id: %d" % (step, sess.run(model.queue_size), FLAGS.task_index))
						except tf.errors.DeadlineExceededError:
							if is_chief and coord.should_stop():
								coord.join(threads, stop_grace_period_secs=10)
								coord.clear_stop()
								sess.run(increment_epoch)
							pass
					if iters>0:
						valid_perplexity = np.exp(costs / iters)
						print("TaskID:%d, Valid epoch: %d  Valid Perplexity: %.4f acc_topk: %f , acc_top1: %f ,speed: %.0f wps"% (FLAGS.task_index, epoch, valid_perplexity, sum(acc_topk_list)/len(acc_topk_list), sum(acc_top1_list)/len(acc_top1_list), iters*FLAGS.batch_size/(time.time()-start_time)))

					current_epoch += 1

					if is_chief:
						if FLAGS.save:
							print("Saving model...")
							sv.saver.save(sess, checkpoint_path, global_step=epoch)
							print("Finish saving model.")		
					
			##test
			test1_costs = 0.0
			test2_costs = 0.0
			test1_acc_list_top3 = []
			test1_acc_list_top1 = []
			test2_acc_list_top3 = []
			test2_acc_list_top1 = []
			f_test_log = open(test_name_log, "w")
			start_time = time.time()
			if is_chief:
				threads = start_threads(test1_data, test1_data_len, test_model, FLAGS.thread_num, pad_id)
				iters = 0
				step = 0
				print("worker_id:%d current_epoch:%d epoch_var:%d"%(FLAGS.task_index, current_epoch, epoch_var.eval()))
				while current_epoch == epoch_var.eval():
					try:
						test_input, test_target, test_seq_len, pred_val, pred_topK, cost_val, step_f_top3,step_f_top1 = sess.run([test_model.input_data, 
								test_model.targets,test_model.seq_len, test_model.pred_val, test_model.pred_topK, test_model.loss, test_model.f_top_k,test_model.f_top_1], options=run_options)
						#pdb.set_trace()
						num_step = (test_input.shape)[-1]
						step_acc3 = cal_dynamic_mean(step_f_top3,test_seq_len,FLAGS.batch_size,num_step)
						step_acc1 = cal_dynamic_mean(step_f_top1,test_seq_len,FLAGS.batch_size,num_step)
						for bs in xrange(FLAGS.batch_size):
							f_test_log.write("# %d %d\n"%(step, num_step))
							for ns in xrange(test_seq_len[bs]):
								f_test_log.write("%s\t"%word_dic[test_input[bs][ns]])
								f_test_log.write("%s\t"%word_dic[test_target[bs][ns]])
								for topk_index in xrange(FLAGS.top_num):
									f_test_log.write("%s\%.4f\t"%(word_dic[pred_topK[bs*num_step + ns][topk_index]], pred_val[bs*num_step +ns][topk_index]))
								f_test_log.write("\n")
							f_test_log.write("\n")
						f_test_log.flush()
						iters += 1
						step += 1
						test1_acc_list_top3.append(step_acc3)
						test1_acc_list_top1.append(step_acc1)
						test1_costs += cost_val
					except tf.errors.DeadlineExceededError:
						if is_chief and coord.should_stop():
							coord.join(threads, stop_grace_period_secs=10)
							coord.clear_stop()
							sess.run(increment_epoch)
						pass
				test1_acc3 = sum(test1_acc_list_top3)/len(test1_acc_list_top3)
				test1_acc1 = sum(test1_acc_list_top1)/len(test1_acc_list_top1)
				if iters> 0:
					test1_ppl = np.exp(test1_costs/iters)
					print("TaskID:%d, Test1 Perplexity: %.4f , acc_topk: %f , acc_top1: %f, speed: %.0f wps"% (FLAGS.task_index, test1_ppl, test1_acc3, test1_acc1, iters*FLAGS.batch_size/(time.time()-start_time)))
				current_epoch += 1

#This func is used to load offline models
def model_load(sess_p,saver,model_index = -1):
	model_path = ""
	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt:
		model_path = ckpt.model_checkpoint_path
		if model_index >= 0 :
			model_path = model_path[:len(model_path)-len(model_path.split("-")[-1])] + str(model_index)
	if model_path!="": 
		#pdb.set_trace()
		dic_file = open(vocab_path, "r")
		#word_dic = dic_file.read().split()
		word_dic = pickle.load(dic_file)
		if word_dic.count("<eos>")==0:
			word_dic.insert(0,"<eos>")
		eos_index = word_dic.index("<eos>")
		if eos_index != 0:
			first_ele = word_dic[0]
			word_dic[0] = "<eos>"
			word_dic[eos_index] = first_ele
		if "<bos>" not in word_dic:
			word_dic.insert(1, "<bos>")
		dic_file.close()
		word_dic = word_dic[:FLAGS.vocab_size]
		word_to_id = dict(zip(word_dic, range(len(word_dic))))
		print ("word_to_id size:%d"%len(word_to_id))
		print("Reading model parameters from %s" % model_path)
		saver.restore(sess_p, model_path)
		return word_to_id,word_dic
	else:
		print("No checkpoint file found!")
		return None

#This func is used to trans word_sequence to id_sequence
def input2id(word_to_id,sen_input):
	input = []
	sen_input = sen_input.split()
	for word in sen_input:
		if not (word_to_id.has_key(word)):
			word = "<unk>"
		input.append(word_to_id.get(word))
	return input

#This func is used to predict the value by input from loaded offline model
def offline_predict(model,sess_p,input,sen_len):
	feed_dict = {}
	feed_dict[model.x] = np.array([input])
	feed_dict[model.x_len] = sen_len
	fetches = model.pred_topK,model.softmax_logits,model.pred_val
	value = sess_p.run(fetches,feed_dict)
	return value

#This func is used for cmd input
def raw_predict(sess_p,model,word_to_id,word_dic):
	sen_input = raw_input("input: ")
	while sen_input:
		input=[]
		print (sen_input)
		sen_input = "<bos> "+sen_input
		input = input2id(word_to_id,sen_input)
		print(input)
		sen_len = len(input)
		if sen_len > 0:
			sen_len = np.array([len(input)])
			values = offline_predict(model,sess_p,input,sen_len)
			#print (values)
			pdb.set_trace()
			value = values[0]
			pdb.set_trace()
			print("top 3 answers are:")
			for ii in xrange(len(value[-1])):
				key = value[-1][ii]
				print("%s" % word_dic[key])
		sen_input = raw_input("input: ")

#This func is used to predict the results of file and calculate the 
def file_predict(sess_p,model,word_to_id,word_dic,file_path,if_long = False):
	test_file = open(file_path)
	total_count = 0.0
	top3_hit = 0.0
	top1_hit = 0.0
	top5_hit = 0.0
	for line in test_file:
		line = line.rstrip()
		line = "<bos> " + line
		line_items = line.split()
		loop_idx_range = range(2,len(line_items))
		if if_long:
			loop_idx_range = range(len(line_items)-1,len(line_items))
		for i in loop_idx_range:
			sen_input = " ".join(line_items[:i])
			print("Input is : "+sen_input)
			pre_word = line_items[i]
			total_count += 1
			print("Expected word is : "+pre_word)
			input = input2id(word_to_id,sen_input)
			sen_len = len(input)
			if sen_len > 0:
				sen_len = np.array([len(input)])
				value = offline_predict(model,sess_p,input,sen_len)					
				#print (value
				word_predict = ""
				for ii in xrange(len(value[-1])):
					key = value[-1][ii]
					word_info = word_dic[key] 
					word_predict += word_info + "#"
					#print("%s" % word_dic[key])
					#top hit calculation
					if word_info == pre_word:
						top5_hit += 1
						if ii < 3:
							top3_hit += 1
						if ii == 0:
							top1_hit += 1
				word_predict = word_predict[:-1] if word_predict!="" else word_predict
				print("Word_predict is : "+word_predict)
	print(file_path + "'s predict acc is >>> top5_acc is : %.2f%%, top3_acc is : %.2f%%, top1_acc is : %.2f%%, top5_hit_count is : %f, top3_hit_count is : %f, top1_hit_count is : %f, total count is : %f."%(top5_hit*100/total_count,top3_hit*100/total_count,top1_hit*100/total_count,top5_hit,top3_hit,top1_hit,total_count))


def predict():
	with tf.Session() as sess:
		predict_op = Option("predict")
		model = RNN_Model(predict_op, reuse=None)
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		wordid2id = np.arange(FLAGS.vocab_size)
		saver = tf.train.Saver(tf.global_variables())
		word_to_id={}
		word_dic = []
		if FLAGS.restore:
			load_results = model_load(sess,saver,9)
			if load_results!=None:
				word_to_id = load_results[0]
				word_dic = load_results[1]
		raw_predict(sess,model,word_to_id,word_dic)
		#file_predict(sess,model,word_to_id,word_dic,test2_path)
		#file_predict(sess,model,word_to_id,word_dic,long_path,True)
				
def main(_):
	if FLAGS.train:
		train()
	if FLAGS.predict:
		predict()

if __name__ == "__main__":
	tf.app.run()

