# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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
from dynamic_lstm import DynamicLSTM
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

flags.DEFINE_bool("restore_model", False, "Restore model")
flags.DEFINE_bool("fetch_model", False, "Fetch model")

FLAGS = flags.FLAGS

def start_threads_func(reader, sess, coord):
	
	def feed_queue_data(data_file, queue):
		data_gen = reader.get_batch_from_file(data_file, queue.batch_size) 
		while not coord.should_stop():
			try:
				inputs, inputs_len, outputs = data_gen.next()
				sess.run(queue.enqueue_op, feed_dict={	queue.inputs: inputs, 
																								queue.inputs_len: inputs_len, 
																								queue.outputs: outputs })
			except StopIteration:
				print("One epoch data is finished")
				# Refresh generator for next epoch
				data_gen = reader.get_batch_from_file(data_file, queue.batch_size)

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
		# Count step for valid epoch
		train_dir = os.path.join(config.data_path, config.dataset, "train")
		valid_dir = os.path.join(config.data_path, config.dataset, "valid")
		total_valid_step = util.count_step(valid_dir, config.batch_size)
		
		# Build graph
		with tf.variable_scope("input"):
			train_queue = Queue("train", config.batch_size)
			valid_queue = Queue("clean", config.batch_size)
	
		with tf.variable_scope("helper"):
			# Define training variables and ops
			do_valid = tf.get_variable("do_valid", shape=[], dtype=tf.bool, initializer=tf.constant_initializer(False), trainable=False)
			train_done = tf.get_variable("train_done", shape=[], dtype=tf.bool, initializer=tf.constant_initializer(False), trainable=False)
			valid_done = tf.get_variable("valid_done", shape=[], dtype=tf.bool, initializer=tf.constant_initializer(False), trainable=False)
			
			temp_val = tf.placeholder(tf.float32, shape=[])
			global_loss = tf.get_variable("global_loss", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)	
			global_loss_update = tf.assign_add(global_loss, temp_val, use_locking=True)
			global_acc_1 = tf.get_variable("global_acc_1", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)	
			global_acc_k = tf.get_variable("global_acc_k", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)	
			global_acc_1_update = tf.assign_add(global_acc_1, temp_val, use_locking=True)
			global_acc_k_update = tf.assign_add(global_acc_k, temp_val, use_locking=True)
	
			global_train_step = tf.get_variable(name='global_train_step', dtype=tf.int32, shape=[], initializer=tf.constant_initializer(0), trainable=False)	
			increment_train_step = tf.assign_add(global_train_step, 1, use_locking=True)
			global_valid_step = tf.get_variable(name='global_valid_step', dtype=tf.int32, shape=[], initializer=tf.constant_initializer(0), trainable=False)	
			increment_valid_step = tf.assign_add(global_valid_step, 1, use_locking=True)
			
				
		with tf.device(tf.train.replica_device_setter(
			worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)): 
			print("Creating %d layers of %d units." % (config.num_layer, config.hidden_size))
			train_model = DynamicLSTM(train_queue, is_training=True, reuse=False)
			valid_model = DynamicLSTM(valid_queue, is_training=False, reuse=True)
	
		model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
		# Add global_train_step to restore if necessary
		model_vars.append(global_train_step)
		model_saver = tf.train.Saver(model_vars, max_to_keep=0)
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		
		with tf.Session(server.target, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
			
			# Create a FileWriter to write summaries
			#summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
		
			sess.run(init_op)		
			print("Variables initialized ...")
			if FLAGS.restore_model:
				ckpt_path = tf.train.latest_checkpoint(config.model_dir)
				if ckpt_path:
					model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
					print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
				else:
					print("model doesn't exists")

			if is_chief:
				if FLAGS.fetch_model:
					model_vals = sess.run(model_vars)
					model_vals_name = ['model_vals/' + i.name.replace("/", "-") for i in model_vars]
					for i in range(len(model_vals)):
						with open(model_vals_name[i], 'wb') as f:
							model_vals[i].tofile(f)
					print("finish fetching model")
				# Create coordinator to control threads
				coord = tf.train.Coordinator()

				# Create start threads function
				start_threads = start_threads_func(reader, sess, coord)
				train_threads = start_threads(train_dir, train_queue)
				valid_threads = start_threads(valid_dir, valid_queue)
				
			valid_ppl_history = []
			while not train_done.eval():
				# Training phase
				while not do_valid.eval():					
					start_time = time.time()
					word_num, train_loss, _ = sess.run([train_model.word_num, train_model.loss, train_model.train_op]) 
					train_speed = word_num // (time.time()-start_time)
					train_ppl = np.exp(train_loss)
					train_step = sess.run(increment_train_step)
					if train_step % config.step_per_log == 0:
						print("TaskID: {} Train step: {} Train ppl: {:.2f} Train speed: {}".format(FLAGS.task_index, train_step, train_ppl, train_speed))
					if train_step % config.step_per_validation == 0:
						sess.run(do_valid.assign(True))
						sess.run(valid_done.assign(False))
				
				# Validation phase
				if is_chief:
					print("start validation")
			
				while global_valid_step.eval() < total_valid_step:
					loss, acc_1, acc_k = sess.run([valid_model.loss, valid_model.top_1_accuracy, valid_model.top_k_accuracy])
					sess.run(increment_valid_step)
					sess.run(global_loss_update, feed_dict={temp_val: loss})
					sess.run(global_acc_1_update, feed_dict={temp_val: acc_1})
					sess.run(global_acc_k_update, feed_dict={temp_val: acc_k})
				
				if is_chief:
					true_valid_step = global_valid_step.eval() 
					valid_loss = sess.run(global_loss)/true_valid_step
					valid_ppl = np.exp(valid_loss)
					valid_acc_1 = sess.run(global_acc_1)/true_valid_step
					valid_acc_k = sess.run(global_acc_k)/true_valid_step
					print("Valid step: {} Valid acc_1: {:.4f}, Valid acc_k: {:.4f} Valid ppl: {:.2f}".format(true_valid_step, valid_acc_1, valid_acc_k, valid_ppl))
						
					# If converged, finish training
					if(len(valid_ppl_history) >= 3 and valid_ppl > max(valid_ppl_history[-3:])):
					#if len(valid_ppl_history) >= 2 and valid_ppl > valid_ppl_history[-1]:
						print("Training is converged")
						sess.run(train_done.assign(True))
					else:
						print("Saving model...")
						# Save graph and model parameters
						model_path = os.path.join(config.model_dir, config.model_name)
						model_saver.save(sess, model_path, global_step=global_train_step.eval())
					valid_ppl_history.append(valid_ppl)
					sess.run(do_valid.assign(False))
					sess.run(valid_done.assign(True))
				else:
					while not valid_done.eval():
						pass
				sess.run(tf.variables_initializer([global_valid_step, global_loss, global_acc_1, global_acc_k]))
				
			
			if is_chief:	
				# Stop data threads
				coord.request_stop()
				# Clean queue before finish
				while train_queue.size.eval() > 0:
					sess.run(train_queue.dequeue_op)
				while valid_queue.size.eval() > 0:
					sess.run(valid_queue.dequeue_op)
				coord.join(train_threads) 
				coord.join(valid_threads)
		
if __name__ == "__main__":
	tf.app.run()

