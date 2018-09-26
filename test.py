# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import time
import pdb
import random
import math
import threading
import pickle
import numpy as np
import tensorflow as tf

import util
import config
from simple_lstm import SimpleLSTM
from reader import Reader 
from queue import Queue

flags = tf.app.flags
logging = tf.logging

# Flags for defining the tf.train.ClusterSpec
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = flags.FLAGS

def start_threads_func(reader, sess, coord):
	
	def feed_queue_data(data_file, queue):
		data_gen = reader.get_batch_from_file(data_file) 
		while not coord.should_stop():
			try:
				inputs, inputs_len, outputs = data_gen.next()
				sess.run(queue.enqueue_op, feed_dict={	queue.inputs: inputs, 
																								queue.inputs_len: inputs_len, 
																								queue.outputs: outputs })
			except StopIteration:
				print("One epoch data is finished")
				# Refresh generator for next epoch
				data_gen = reader.get_batch_from_file(data_file)

	def start_threads(data_dir, queue):
		# Create fresh generator every time you call start_threads
		data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
		thread_num = len(data_files)
		threads = []
		# create one thread for reading one single file
		for i in range(thread_num):
			t = threading.Thread(target=feed_queue_data, args=(data_files[i], queue))
			threads.append(t)
		for thread in threads:
			thread.daemon = True
			thread.start()
		return threads
	
	return start_threads

def restore_model(sess, model_saver):
	ckpt_path = tf.train.latest_checkpoint(config.model_dir)
	if ckpt_path:
		model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
		print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
	else:
		print("model doesn't exists")

def main(_):
	ps_hosts = FLAGS.ps_hosts.split(",")
	worker_hosts = FLAGS.worker_hosts.split(",")
	worker_num = len(worker_hosts)
				
	# Create a cluster from the parameter server and worker hosts.
	cluster = tf.train.ClusterSpec({ "ps": ps_hosts, "worker" : worker_hosts })
	
	# Start a server for a specific task
	server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
	
	#print("I'm worker %d and my server target is, "%FLAGS.task_index, server.target)	
	if FLAGS.job_name == "ps":
		server.join()
	elif FLAGS.job_name == "worker":
		is_chief = FLAGS.task_index == 0	
		if is_chief:
			# Create reader for reading raw data	
			reader = Reader()
		# Count step for test epoch
		test_dir = os.path.join(config.data_path, config.dataset, "test_sorted")
		total_test_step = util.count_step(test_dir, config.batch_size)
		
		with tf.device("/job:ps/cpu:0"):
			with tf.variable_scope("input"):
				test_queue = Queue("test", config.batch_size)
	
			with tf.variable_scope("helper"):
				# Define training variables and ops
				temp_val = tf.placeholder(tf.float32, shape=[])
				global_loss = tf.get_variable("global_loss", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)	
				global_loss_update = tf.assign_add(global_loss, temp_val, use_locking=True)
				global_acc = tf.get_variable("global_acc", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)	
				global_acc_update = tf.assign_add(global_acc, temp_val, use_locking=True)
	
				global_test_step = tf.get_variable(name='global_test_step', dtype=tf.int32, shape=[], initializer=tf.constant_initializer(0), trainable=False)	
				increment_test_step = tf.assign_add(global_test_step, 1, use_locking=True)
			
				
		with tf.device(tf.train.replica_device_setter(
			worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)): 
			print("Creating %d layers of %d units." % (config.num_layer, config.hidden_size))
			test_model = SimpleLSTM(test_queue, is_training=False, reuse=False)
	
		model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
		model_saver = tf.train.Saver(model_vars)
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	# Test Phase
	with tf.Session(server.target, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		
		print("Variables initialized ...")
		sess.run(init_op)
		restore_model(sess, model_saver)				
		
		if is_chief:
			# Create coordinator to control threads
			coord = tf.train.Coordinator()
	
			# Create start threads function
			start_threads = start_threads_func(reader, sess, coord)
			test_threads = start_threads(test_dir, test_queue)
			
		while global_test_step.eval() < total_test_step:
			loss, acc = sess.run([test_model.loss, test_model.accuracy])
			sess.run(increment_test_step)
			sess.run(global_loss_update, feed_dict={temp_val: loss})
			sess.run(global_acc_update, feed_dict={temp_val: acc})

		if is_chief:
			# Stop data threads
			coord.request_stop()
			# Clean queue before finish
			while test_queue.size.eval() > 0:
				sess.run(test_queue.dequeue_op)

			coord.join(test_threads)
			true_test_step = global_test_step.eval()
			test_loss = sess.run(global_loss)/true_test_step
			test_ppl = np.exp(test_loss)
			test_acc = sess.run(global_acc)/true_test_step
			print("Test step: {} Test ppl: {:.2f} Test acc: {:.2f}".format(true_test_step, test_ppl, test_acc))

if __name__ == "__main__":
	tf.app.run()

