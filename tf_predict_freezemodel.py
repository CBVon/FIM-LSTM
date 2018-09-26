import argparse 
from reader import Reader
from tf_load_graph import load_graph
import tensorflow as tf

if __name__ == '__main__':
	# Let's allow the user to pass the filename as an argument
	parser = argparse.ArgumentParser()
	parser.add_argument("--frozen_model_filename", default="./model/dynamic_lstm_1layer_embed96_hid256_share_model_newdic.pb", type=str, help="Frozen model file to import")
	args = parser.parse_args()

	# We use our "load_graph" function
	graph = load_graph(args.frozen_model_filename)

	# We can verify that we can access the list of operations in the graph
	for op in graph.get_operations():
		 print(op.name)
		# prefix/Placeholder/inputs_placeholder
		# ...
		# prefix/Accuracy/predictions
		
	# We access the input and output nodes 
	x = graph.get_tensor_by_name('prefix/model/input/x:0')
	x_len = graph.get_tensor_by_name('prefix/model/input/x_len:0')
	y = graph.get_tensor_by_name('prefix/model/predict/Softmax:0')
		
	# We launch a Session
	import time
	reader = Reader()
	with tf.Session(graph=graph) as sess:
		# Note: we didn't initialize/restore anything, everything is stored in the graph_def
		user_input = raw_input("input: ")
		while user_input:
			start = time.clock()
			inputs, inputs_len, outputs = reader.get_batch_from_input(user_input)
			feed_dict={x: inputs, x_len: inputs_len}
			prob = sess.run(y, feed_dict=feed_dict)
			elapsed = (time.clock() - start)
			print("Time used:",elapsed)
			#y_out = sess.run(y, feed_dict={
		  	#  x: [[3, 5, 7]],
					#	x_len:[3]
			#})
			print(prob) # [[ False ]] Yay, it works!
			user_input = raw_input("input: ")
