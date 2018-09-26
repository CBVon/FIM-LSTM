# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pdb
import random
import math
import pickle
import numpy as np
import tensorflow as tf
import config
#from it_corpus2_es96_hs512 import config

class DynamicLSTM(object):
	def __init__(self, queue, is_training, reuse=None):
		with tf.variable_scope("model", reuse=reuse):
			with tf.name_scope("input"):
				if queue:
					self.x, self.x_len, self.y = queue.dequeue_op
				else:
					self.x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x')
					self.x_len = tf.placeholder(dtype=tf.int32, shape=[None], name='x_len')
					self.y = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y')
				
				batch_size = tf.shape(self.x)[0]	
				padding_len = tf.shape(self.x)[1]
				self.word_num = tf.reduce_sum(self.x_len)
			
			with tf.name_scope("embedding"):
				#TODO try xavier initialization
				#embedding = tf.get_variable("embedding", [config.vocab_size, config.embedding_size])
				#变频Embedding部分
				high_embedding = tf.get_variable("high_embedding", [int(0.2 * config.vocab_size), config.embedding_size])
				low_embedding = tf.get_variable("low_embedding", [config.vocab_size - int(0.2 * config.vocab_size), config.embedding_size / 3])
				
				#fc 对齐 	
				low_to_high_fc = tf.get_variable("low_to_high_fc", [config.embedding_size / 3, config.embedding_size])
				embedding = tf.concat([high_embedding, tf.matmul(low_embedding, low_to_high_fc)], 0)  
				
				#copy 对齐 #3个单位阵 水平拼接
				#low_to_high_copy = tf.constant([[1.0 if j % 128 == i else 0.0 for j in xrange(384)] for i in xrange(128)])
				#embedding = tf.concat([high_embedding, tf.matmul(low_embedding, low_to_high_copy)], 0)
				
				'''
				#无效： assert isinstance(grad, ops.Tensor) tile无法求导
				low_to_high_tile = tf.tile(low_embedding, [1, 3])
				embedding = tf.concat([high_embedding, low_to_high_tile], 0)
				embedding = tf.cast(embedding, tf.float32)
				'''

				inputs = tf.nn.embedding_lookup(embedding, self.x)
				if is_training:
					inputs = tf.nn.dropout(inputs, config.input_keep_prob)
				#inputs = tf.nn.embedding_lookup(embedding, self.x)
				self.inputs = inputs	
		
			with tf.name_scope("rnn"):	
				def lstm_cell():
					lstm_cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, reuse=tf.get_variable_scope().reuse)
					lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, 
																										output_keep_prob=config.output_keep_prob
																									 ) if is_training else lstm_cell
					return lstm_cell

				cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(config.num_layer)])
				
				outputs, last_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=self.x_len, dtype=tf.float32)
				self.last_state = last_state[0].h	
				mask = tf.sequence_mask(self.x_len, padding_len)
				outputs_flat = tf.boolean_mask(outputs, mask)
				y_flat = tf.boolean_mask(self.y, mask)
				#softmax_w = tf.get_variable("softmax_w", [config.vocab_size, config.embedding_size])	
				#trans_matrix = tf.get_variable("trans_matrix", [config.embedding_size, config.hidden_size])
				softmax_w = tf.get_variable("softmax_w", [config.hidden_size, config.embedding_size])
				softmax_b = tf.get_variable("softmax_b", [config.embedding_size])
				#softmax_w = tf.matmul(embedding, trans_matrix)
				#softmax_b = tf.get_variable("softmax_b", [config.vocab_size])
			
				#logits_flat = tf.nn.bias_add(tf.matmul(outputs_flat, softmax_w, transpose_a=False, transpose_b=True), softmax_b) 
				logits_flat = tf.nn.bias_add(tf.matmul(outputs_flat, softmax_w, transpose_a=False, transpose_b=False), softmax_b)
				logits_flat = tf.matmul(logits_flat, embedding, transpose_a=False, transpose_b=True)
					
			with tf.name_scope("loss"):
				self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
								 			labels = y_flat,
								 			logits = logits_flat,
								 			name = "loss"))

			with tf.name_scope("train"):
				optimizer = tf.train.AdamOptimizer(use_locking=True)
				tvars = tf.trainable_variables()
				grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.max_grad_norm)
				self.train_op = optimizer.apply_gradients(zip(grads, tvars))
				
			with tf.name_scope("accuracy"):
				#outputs_prob_flat = tf.nn.softmax(logits_flat)
				#is_top_k = tf.nn.in_top_k(outputs_prob_flat, y_flat, config.top_k)  
				# softmax is slow here so we use logits
				is_top_1 = tf.nn.in_top_k(logits_flat, y_flat, 1)
				is_top_k = tf.nn.in_top_k(logits_flat, y_flat, config.top_k)  
				self.top_1_accuracy = tf.reduce_mean(tf.cast(is_top_1, tf.float32))
				self.top_k_accuracy = tf.reduce_mean(tf.cast(is_top_k, tf.float32))
		
			with tf.name_scope("predict"):
				#self.last_output = tf.nn.bias_add(tf.matmul(last_state[0].h, softmax_w, transpose_a=False, transpose_b=True), softmax_b)
				self.last_output = tf.nn.bias_add(tf.matmul(last_state[0].h, softmax_w, transpose_a=False, transpose_b=False), softmax_b)
				self.last_output = tf.matmul(self.last_output, embedding, transpose_a=False, transpose_b=True)
				#pdb.set_trace()
				self.output_prob = tf.nn.softmax(self.last_output)
				self.top_k_prob, self.top_k_ind = tf.nn.top_k(logits_flat, config.top_k)
