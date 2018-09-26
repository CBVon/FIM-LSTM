# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import takewhile, repeat
import multiprocessing
import pickle
import config

#from it_corpus2_es96_hs512 import config

class Reader(object):
	def __init__(self):
		self.BOS = "<s>"
		self.EOS = "</s>"
		self.UNK = "<unk>"
		self.vocab_size = config.vocab_size

		self.build_vocab()
	
	def build_vocab(self):
		#vocab_path = os.path.join(config.data_path, config.dataset, "vocabulary.txt")
		vocab_path = os.path.join(config.data_path, config.dataset, "vocabulary.pkl")
		#high_vocab_path = os.path.join(config.data_path, config.dataset, "vocabulary_high.pkl") #top20
		#low_vocab_path = os.path.join(config.data_path, config.dataset, "vocabulary_low.pkl") #low80
		#vocab_path = ('./id_corpus4.pkl')
		#train_dir = os.path.join(config.data_path, config.dataset, "train")
		valid_dir = os.path.join(config.data_path, config.dataset, "valid")
		#test_dir = os.path.join(config.data_path, config.dataset, "test")
		
		#if os.path.isfile(vocab_path) and os.path.isfile(high_vocab_path) and os.path.isfile(low_vocab_path):
		if os.path.isfile(vocab_path):
			#words = open(vocab_path).read().replace("\n", " ").split()
			vocab_file = open(vocab_path, 'rb')
			words = pickle.load(vocab_file)
			if words.count(self.BOS)==0:
				words.insert(0,self.BOS)
			else:
				bos_idx = word.index(self.BOS)
				del words[bos_idx]
				words.insert(0, self.BOS)
			
			if words.count(self.EOS)==0:
				words.insert(0,self.EOS)
			else:
				eos_idx = word.index(self.EOS)
				del words[eos_idx]
				words.insert(0, self.EOS)

			if words.count(self.UNK)==0:
				words.insert(0,self.UNK)
			else:
				unk_idx = words.index(self.UNK)
				del words[unk_idx]
				words.insert(0,self.UNK)
				'''
				#不能swap；为了保持频率top→low，需要del+insert
				if unk_idx != 0:
					ori_ele = words[0]
					words[0] = self.UNK
					words[unk_idx] = ori_ele
				'''
			self.words = words[: config.vocab_size]
			#print(len(self.words))
			assert len(self.words) == config.vocab_size
			vocab_file.close()
		else:#重构词表
			data = []
			#train_files = os.listdir(train_dir)
			#for f in [os.path.join(train_dir, train_file) for train_file in train_files]:
			#	data.extend(self.read_words(f))
			valid_files = os.listdir(valid_dir)
			for f in [os.path.join(valid_dir, valid_file) for valid_file in valid_files]:
				data.extend(self.read_words(f))
			#test_files = os.listdir(test_dir)
			#for f in [os.path.join(test_dir, test_file) for test_file in test_files]:
			#	data.extend(self.read_words(f))
			
			counter = Counter(data)  # sort words '.': 5, ',': 4......
			count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
			words = list(zip(*count_pairs)[0])
			#print("total vocab size: %d"%len(words))
			
			# make sure <unk> is in vocabulary
			if self.UNK not in words:
				words.insert(0, self.UNK)
			# make sure EOS is id 0
			#words.insert(0, self.EOS)

			# Save the vocabulary with pickle for future use
			#vocab_file = open(vocab_path, 'wb')
			#pickle.dump(words[:10000], vocab_file)
			pickle.dump(words[: config.vocab_size], open(vocab_path, 'wb')) #20180627
			#pickle.dump(words[: int(0.2 * config.vocab_size)], open(high_vocab_path, "wb")) #20180701
			#pickle.dump(words[int(0.2 * config.vocab_size): config.vocab_size], open(low_vocab_path, "wb"))
			#vocab_file.close()
		self.words = words[: config.vocab_size]
		#self.high_words = words[: int(0.2 * config.vocab_size)]
		#self.low_words = words[int(0.2 * config.vocab_size): config.vocab_size]
	  #	print("words size: %d")
		#assert len(self.words) == config.vocab_size
		#print("vocab size: %d" % (len(self.high_words) + len(self.low_words)))
		print("vocab size: %d" % len(self.words))
		self.word2id = dict(zip(self.words, range(config.vocab_size))) 
	
	def read_words(self, file_path):  # return 1-D list
		words = []
		with open(file_path) as f:
			for line in f:
				words.extend(line.strip().split())
		return words

	def from_line_to_id(self, line):
		word_list = line.strip().split()
		#word_list.append(self.EOS)
		id_list = [self.word2id[word] if self.word2id.has_key(word) else self.word2id[self.UNK] for word in word_list]
		#id_list = [self.word2id[word] for word in word_list]
		return id_list

	def padding_batch(self, batch_data):
		batch_size = len(batch_data)
		padding_len = len(max(batch_data, key=len))
		result = np.zeros([batch_size, padding_len])
		result_len = np.zeros(batch_size)
		for i, data in enumerate(batch_data):
			data_len = len(data)
			result_len[i] = data_len 
			result[i][:data_len] = data
		return result, result_len
	
	def get_batch_from_file(self, data_file, batch_size):
		batch_inputs = []
		batch_outputs = []
		with open(data_file, 'r') as f:
			for line in f:
				id_list = self.from_line_to_id(line)
				batch_inputs.append(id_list[:-1])
				batch_outputs.append(id_list[1:])
				if len(batch_outputs) == batch_size:
					inputs, inputs_len = self.padding_batch(batch_inputs)
					outputs, _ = self.padding_batch(batch_outputs)
					batch_inputs = []
					batch_outputs = []
					yield inputs, inputs_len, outputs 
	
	def get_custom_line_from_file(self, data_file):
		with open(data_file, 'r') as f:
			for line in f:
				id_list = self.from_line_to_id(line)
				if len(id_list) <= config.max_len:
					inputs_len = np.array(len(id_list)).reshape([-1])
					id_list.extend([0] * (config.max_len - len(id_list)))
				else:
					inputs_len = np.array(config.max_len).reshape([-1])
					id_list = id_list[-config.max_len:]
				inputs = np.array(id_list).reshape([-1, config.max_len])
				yield inputs, inputs_len 

	def get_custom_batch_from_file(self, data_file, batch_size):
		batch_inputs = []
		batch_outputs = []
		with open(data_file, 'r') as f:
			for line in f:
				word_list = line.strip().split()
				input_list = [self.word2id[word] if self.word2id.has_key(word) else self.word2id[self.UNK]for word in word_list[:3]]
				output_list = [self.word2id[word] if self.word2id.has_key(word) else self.word2id[self.UNK] for word in word_list[4:]]
				#output_list = [self.word2id[word] for word in word_list[4:7]]
				#input_list = [self.word2id[word_list[0]] if self.word2id.has_key(word_list[0]) else self.word2id[self.UNK]]
				#output_list = [self.word2id[word] for word in word_list[3:6]]
				batch_inputs.append(input_list)
				batch_outputs.append(output_list)
				if len(batch_outputs) == batch_size:
					inputs = np.array(batch_inputs)
					outputs = np.array(batch_outputs)
					inputs_len = np.ones(batch_size) * 3
					batch_inputs = []
					batch_outputs = []
					yield inputs, inputs_len, outputs 

	def get_batch_from_input(self, line):
		id_list = self.from_line_to_id(line)
		#id_list = [1, 3]
		inputs = np.array(id_list).reshape((1,-1))
		print (inputs)
		inputs_len = np.array([len(id_list)])
		outputs = np.array(id_list).reshape((1,-1))
		return inputs, inputs_len, outputs
