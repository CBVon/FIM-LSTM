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
from collections import defaultdict
from string import punctuation
from dynamic_lstm import DynamicLSTM
from reader import Reader 
from queue import Queue
import config
#from it_corpus2_es96_hs512 import config
#from batch19_es96_hs512 import config
#This func aims to build a header dict.
def words_2_map(reader):
	header_map=defaultdict(list)
	words_total = reader.words
	words2id = reader.word2id
	for word in words_total:
		if not words2id.has_key(word):
			sys.stderr.write("Word "+word + " not existed in word2id dict\n")
			return {}
		else:
			wordid = words2id[word]
		for i in range(len(word)):
			header = word[:(i+1)]
			header_map[header].append(wordid)
	return header_map

def get_header_prob(header_map,last_prob,input_header,topk):
		header_ids = header_map[input_header]
		top_ind = []
		prob_id_sorted = last_prob.argsort()[::-1]
		top_idx = 0
		for id_tmp in prob_id_sorted:
			if header_ids.count(id_tmp)>0:
				top_ind.append(id_tmp)
				top_idx += 1
			if top_idx >= topk:
				break
		return top_ind

def test_from_input():
	reader = Reader()
	header_map = words_2_map(reader)
	if len(header_map)==0:
		return
	model = DynamicLSTM(None, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	#output_graph = "./model/dynamic_lstm_1layer_embed96_hid256_share_model_newdic.pb"
	model_saver = tf.train.Saver(model_variables)
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		#ckpt_path = tf.train.latest_checkpoint('dy_share_lstm_batch13')
		print(config.model_dir)
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		print(ckpt_path)
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
		sess.run(init_op)
		if ckpt_path:
			model_saver.restore(sess, ckpt_path)
			print("Read model parameters from %s" % ckpt_path)
		else:
			print("model doesn't exists")
		user_context = raw_input("Context: ").strip()
		input_header = raw_input("User Input : ").strip()
		while user_context:
			inputs, inputs_len, outputs = reader.get_batch_from_input(user_context)
			feed_dict={ model.x: inputs, model.x_len: inputs_len}
			prob = sess.run([model.output_prob], feed_dict=feed_dict)
			#prob, last_state, embed, outputsflat = sess.run([model.output_prob, model.last_state, model.inputs, model.outputs_flat], feed_dict=feed_dict)
			last_prob = prob[-1][-1]
			#pdb.set_trace()
			if not input_header or not header_map.has_key(input_header):
				top3_ind = last_prob.argsort()[-3:][::-1]
			else:
				top3_ind = get_header_prob(header_map,last_prob,input_header,3)
			sentence = []
			user_input_words = user_context.split()
			print("input: " + " ".join(user_context))
			print("top answers are:")
			words = [reader.words[i] for i in top3_ind]
			print(" ".join(words))
			user_context = raw_input("input: ").strip()
			input_header = raw_input("User Input : ").strip()

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

#Punc is not the target word in this function,  as punc hit is 100% when given a uni input.
def ajust_eval(filepath,loop_idx = -1):
	#Init reader and load model
	reader = Reader()
	header_map = words_2_map(reader)
	if len(header_map)==0:
		return
	model = DynamicLSTM(None, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_variables)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		sess.run(init_op)
		#Model load
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		if loop_idx > 0:
			ckpt_path = ckpt_path.split('-')[0] + '-' + str(loop_idx)
		if ckpt_path:
			model_saver.restore(sess, ckpt_path)
			print("Read model parameters from %s" % ckpt_path)
		else:
			print("model doesn't exists")

		#File handle
		f = open(filepath)
		click_total = {"total":0,1:0,3:0,5:0}
		hit_total = {"total":0,1:0,3:0,5:0}
		UPPERS=['NUM','TELEPHONE','DIGIT','SCORES','USER','YEARGAP']

		#File loop, file must be format context \t(input\t)word
		for line in f:
			line = line.strip()
			line_items = line.split('\t')
			if len(line_items)<2:
				sys.stderr.write("Error line format, items missing of line: "+line+'\n')
			#line info handle
			context = line_items[0].strip()
			word = line_items[-1].strip()
			input_str = word
			if len(line_items)==3:
				input_str = line_items[1].strip()
			if input_str == "" or punctuation.count(word)>0 or word.isdigit() or UPPERS.count(word)>0:
				continue

			#Data model input output
			inputs, inputs_len, outputs = reader.get_batch_from_input(context)
			feed_dict={ model.x: inputs, model.x_len: inputs_len}
			prob = sess.run([model.output_prob], feed_dict=feed_dict)
			last_prob = prob[-1][-1]

			#Exp starts
			click_total["total"] += len(word)#Click rate's denominator
			hit_total["total"] += 1
			input_len = len(input_str)
			tmp_click_count = {1:input_len,3:input_len,5:input_len}#list to record the click count for top1, top3, top5
			tmp_hit_count = {1:0,3:0,5:0}
			for i in range(1,(len(input_str)+1)):
				input_header = input_str[:i]
				top5_ind = get_header_prob(header_map,last_prob,input_header,5)
				words = [reader.words[j] for j in top5_ind]
				if len(words)==0:
					words.append(input_header)
				for idx in [1,3,5]:
					if words[:idx].count(word)>0:
						if tmp_click_count[idx]==input_len:
							tmp_click_count[idx] = i
						if i == 1:
							tmp_hit_count[idx] = 1
				print("Context is : "+context)
				print("Input is : "+input_header)
				print("Expected word is : " + word)
				print("Word_predict is : "+"#".join(words))
				if words[:1].count(word)>0:
					break
			for idx in [1,3,5]:
				click_total[idx] += tmp_click_count[idx]
				hit_total[idx] += tmp_hit_count[idx]
		
		print(filepath + "'s type-rate is >>> top5_type-rate is : %.2f%%, top3_type-rate is : %.2f%%, top1_type-rate is : %.2f%%, top5_type_total is : %f, top3_type_total is : %f, top1_type_total is : %f, total type count is : %f."%(click_total[5]*100/click_total['total'], click_total[3]*100/click_total['total'],click_total[1]*100/click_total['total'],click_total[5], click_total[3],click_total[1],click_total['total']))
		print(filepath + "'s uni-input-hit-rate is >>> top5_hit_rate is : %.2f%%, top3_hit_rate is : %.2f%%, top1_hit_rate is : %.2f%%, top5_hit_total is %f,top3_hit_total is %f,top1_hit_total is %f,total count is : %f."%(hit_total[5]*100/hit_total['total'], hit_total[3]*100/hit_total['total'],hit_total[1]*100/hit_total['total'],hit_total[5], hit_total[3],hit_total[1],hit_total['total']))

def main(_): 
	if len(sys.argv) > 1:
		#test_from_file(sys.argv[1])
		test_top_3(sys.argv[1],111602,True)
		#ajust_eval(sys.argv[1],111602)
		#ajust_eval(sys.argv[1],-1)
	else:
		test_from_input()
	#test_from_file(sys.argv[1])
	#test_top_3(sys.argv[1]) 

if __name__ == "__main__":
	tf.app.run()
