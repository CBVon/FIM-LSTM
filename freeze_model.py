import tensorflow as tf
from tensorflow.python.framework import graph_util

flags = tf.app.flags
FILEDIR = 'dy_share_lstm_batch13'

def data_type():
	return tf.float32
	
def freeze_graph(sess, model_folder):
	checkpoint = tf.train.get_checkpoint_state(model_folder)
	input_checkpoint = checkpoint.model_checkpoint_path
	# We precise the file fullname of our freezed graph  
	output_graph = model_folder + "/dynamic_lstm_1layer_embed96_hid512_share_model_batch13.pb"

	print (output_graph)
	output_node_names = 'model/predict/Softmax,model/rnn/c_out,model/rnn/h_out'
	# We import the meta graph and retrive a Saver  
	saver_meta = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices = True)  
	# We retrieve the protobuf graph definition  
	graph = tf.get_default_graph()  
	tf.train.write_graph(graph, FILEDIR, 'nimei_dynimic.pb',as_text=True)
	#tf.train.write_graph(graph,'score_model','fuck1.pb',as_text=True)
	input_graph_def = graph.as_graph_def()  
	#We start a session and restore the graph weights  
	print ('ckpt:',input_checkpoint)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)
	saver_meta.restore(sess, input_checkpoint)
	output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(','))
	#print(output_graph_def)
	with tf.gfile.GFile(output_graph, "wb") as f:
		f.write(output_graph_def.SerializeToString())  
	print("%d ops in the final graph." % len(output_graph_def.node))

#saver = tf.train.Saver()
#a = tf.get_variable("RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias_1",shape=())
with tf.device('/cpu:0') as device:
	with tf.Session() as sess:
		#saver.restore(sess,save_path)
		#saver = tf.train.import_meta_graph(save_path+'.meta')  
		#sess.run(init_op)
		#sess.run(a.initializer)
		freeze_graph(sess, FILEDIR)
