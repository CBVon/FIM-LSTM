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
import pickle
import argparse
import numpy as np
import tensorflow as tf

from dynamic_lstm import DynamicLSTM
from reader import Reader 
from queue import Queue
import config

def fetch_model(sess,model_variables):
	model_vals_name = []
	for i in range(5):
		model_val_name = model_variables[i]
		model_vals_name.append("model_vals_dlstm_dbg_uni/"+model_val_name.name.replace("/", "-"))
	model_variables = sess.run(model_variables)
	for i in range(5):
		with open(model_vals_name[i], 'wb') as f:
			model_variable = model_variables[i]
			model_shape = model_variable.shape
			if model_shape == ((config.embedding_size + config.hidden_size), (4*config.hidden_size)):
				print(model_vals_name[i])
				pdb.set_trace()
				model_variable = model_variable.T
			if model_shape == ((config.hidden_size,config.embedding_size)):
				print(model_vals_name[i])
				model_variable = model_variable.T
			print(model_variable.shape)
			model_variable.tofile(f)
	print("finish fetching model")

def test_from_input():
	reader = Reader()
	model = DynamicLSTM(None, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	#pdb.set_trace()
	#output_graph = "./model/dynamic_lstm_1layer_embed96_hid256_share_model_newdic.pb"
	model_saver = tf.train.Saver(model_variables)
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
		sess.run(init_op)
		if ckpt_path:
			model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
		else:
			print("model doesn't exists")
		#fetch_model(sess,model_variables)
		user_input = raw_input("input: ")
		while user_input:
			inputs, inputs_len, outputs = reader.get_batch_from_input(user_input)
			feed_dict={ model.x: inputs, model.x_len: inputs_len}
			prob = sess.run([model.output_prob], feed_dict=feed_dict)
			pdb.set_trace()
			top3_ind = prob[-1][-1].argsort()[-10:][::-1]
			sentence = []
			user_input_words = user_input.split()
			print("input: " + " ".join(user_input))
			print("top answers are:")
			words = [reader.words[i] for i in top3_ind]
			print(" ".join(words))
			user_input = raw_input("input: ")

def test_from_file(filepath):
	reader = Reader()
	model = DynamicLSTM(None, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_variables)
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		if ckpt_path:
			model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
		else:
			print("model doesn't exists")
		model_path = os.path.join(config.model_dir, config.model_name)
		model_saver.save(sess, model_path, global_step=0)
		data_gen = reader.get_custom_line_from_file(filepath)
		for inputs, inputs_len in data_gen:
			feed_dict={	model.x: inputs, model.x_len: inputs_len}
			prob = sess.run([model.output_prob], feed_dict=feed_dict)


def test_top_3(filepath,loop_idx = -1,if_long = False):
	reader = Reader()
	model = DynamicLSTM(None, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_variables)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		sess.run(init_op)
		
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		if loop_idx > 0:
			ckpt_path = ckpt_path.split('-')[0] + '-' + str(loop_idx)
		if ckpt_path:
			model_saver.restore(sess, ckpt_path)
			print("Read model parameters from %s" % ckpt_path)
		else:
			print("model doesn't exists")
		f = open(filepath)
		total_count = 0.0
		top5_hit = 0.0
		top3_hit = 0.0
		top1_hit = 0.0
		for line in f:
			line = line.strip()
			line_items = line.split()
			line_len = len(line_items)
			if line_len == 1:
				continue
			beg_idx = 1
			if if_long:
				beg_idx = line_len - 1
			for i in range(beg_idx,line_len):
				total_count += 1
				context = " ".join(line_items[:i])
				word = line_items[i]
				inputs, inputs_len, outputs = reader.get_batch_from_input(context)
				feed_dict={ model.x: inputs, model.x_len: inputs_len}
				prob = sess.run([model.output_prob], feed_dict=feed_dict)
				top5_ind = prob[-1][-1].argsort()[-5:][::-1]
				words = [reader.words[i] for i in top5_ind]
				if words[0]==word:
					top1_hit += 1
				if words[:3].count(word)>0:
					top3_hit += 1
				if words[:5].count(word)>0:
					top5_hit += 1
				print("Input is : "+context)
				print("Expected word is : " + word)
				print("Word_predict is : "+"#".join(words))
		print(filepath + "'s predict acc is >>> top5_acc is : %.2f%%, top3_acc is : %.2f%%, top1_acc is : %.2f%%, top5_hit_count is : %f, top3_hit_count is : %f, top1_hit_count is : %f, total count is : %f."%(top5_hit*100/total_count, top3_hit*100/total_count,top1_hit*100/total_count,top5_hit, top3_hit,top1_hit,total_count))

def main(_):
	if len(sys.argv) > 1:
		#test_from_file(sys.argv[1])
		test_top_3(sys.argv[1],111602,True)
	else:
		test_from_input()
	#test_from_file(sys.argv[1])
	#test_top_3(sys.argv[1])

if __name__ == "__main__":
	tf.app.run()
