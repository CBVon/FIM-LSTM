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
def test_from_input():
	reader = Reader()
	model = DynamicLSTM(None, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	output_graph = "./model/dynamic_lstm_1layer_embed96_hid256_share_model_newdic.pb"
	model_saver = tf.train.Saver(model_variables)
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		output_node_names = 'model/predict/Softmax'
		#ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		#ckpt_path = tf.train.get_checkpoint_state(config.model_dir)
		checkpoint = tf.train.get_checkpoint_state(config.model_dir)
		input_checkpoint = checkpoint.model_checkpoint_path
		saver_meta = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices = True)  
		graph = tf.get_default_graph()
		tf.train.write_graph(graph, FILEDIR, 'dynamic_lstm_1layer_embed96_hid256_share_model_newdic_meta.pb',as_text=True)
		input_graph_def = graph.as_graph_def()
		print ('ckpt:', input_checkpoint)
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
		sess.run(init_op)
		saver_meta.restore(sess, input_checkpoint)
		output_graph_def = graph_util.convert_variables_to_constants(sess,input_graph_def, output_node_names.split(","))
		tf.train.write_graph(output_graph_def,FILEDIR,'fuck2_accuracy.pb',as_text=True)
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(output_graph_def.SerializeToString())
		print("%d ops in the final graph." % len(output_graph_def.node))
		if ckpt_path:
			model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
		else:
			print("model doesn't exists")
		freeze_graph(sess, 'model')
		#model_path = os.path.join(config.model_dir, config.model_name)
		#model_saver.save(sess, model_path, global_step=0)
		user_input = raw_input("input: ")
		while user_input:
			inputs, inputs_len, outputs = reader.get_batch_from_input(user_input)
			feed_dict={ model.x: inputs, model.x_len: inputs_len}
			prob = sess.run([model.output_prob], feed_dict=feed_dict)
			top3_ind = prob[-1][-1].argsort()[-3:][::-1]
			sentence = []
			user_input_words = user_input.split()
			#for w in range(len(user_input_words)):
			#import pdb
			#pdb.set_trace()
			#sentence.append(user_input_words[w])
			print("input: " + " ".join(user_input))
			#print("input: " + " ".join(sentence))
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


def test_top_3(filepath):
	reader = Reader()
	queue = Queue("test", config.batch_size)
	model = DynamicLSTM(queue, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_variables)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		sess.run(init_op)
		
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		if ckpt_path:
			model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
		else:
			print("model doesn't exists")
			
		data_gen = reader.get_custom_batch_from_file(filepath, config.batch_size)
		correct = 0
		total = 0
		line_num = 0
		for inputs, inputs_len, outputs in data_gen:
			line_num += 1
			sess.run(queue.enqueue_op, feed_dict={	queue.inputs: inputs, 
																							queue.inputs_len: inputs_len, 
																							queue.outputs: outputs })
			x, cands, last_state, last_output, probs = sess.run([model.x, model.y, model.last_state, model.last_output, model.output_prob])
			for i in range(config.batch_size):
				#pdb.set_trace()
				for word_ind in range(3):
					print("%s "%reader.words[x[i][word_ind]], end=' ')
				max_val = probs[i][cands[i][0]]
				#if max_val > probs[i][cands[i][1]] and max_val > probs[i][cands[i][2]] and max_val > probs[i][cands[i][3]] and max_val > probs[i][cands[i][4]] and max_val > probs[i][cands[i][5]] and max_val > probs[i][cands[i][6]] and max_val > probs[i][cands[i][7]] and max_val > probs[i][cands[i][8]] and max_val > probs[i][cands[i][9]]:
				#if max_val > probs[i][cands[i][1]] and max_val > probs[i][cands[i][2]] and max_val > probs[i][cands[i][3]] and max_val > probs[i][cands[i][4]]:
				if max_val > probs[i][cands[i][1]] and max_val > probs[i][cands[i][2]]:
					correct += 1 
					print("correct ", end='')
				else:
					print("incorrect ", end='')
				for word_ind in range(10):
					print("%s(%f) "%(reader.words[cands[i][word_ind]], probs[i][cands[i][word_ind]]), end=' ')
				print("\n")
			total += config.batch_size
			#print("total: %d correct: %d correct rate: %.4f" %(total, correct, float(correct)/float(total)))
		print("total: %d correct: %d correct rate: %.4f" %(total, correct, float(correct)/float(total)))



def main(_):
	if len(sys.argv) > 1:
		test_from_file(sys.argv[1])
	else:
		test_from_input()
	#test_from_file(sys.argv[1])
	#test_top_3(sys.argv[1])

if __name__ == "__main__":
	tf.app.run()
