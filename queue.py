import tensorflow as tf

class Queue(object):
	def __init__(self, name, batch_size):	
		# Place queue on parameter server.
		self.inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, None], name="inputs")
		self.inputs_len = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="inputs_len")
		self.outputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, None], name="outputs")
		
		with tf.container("queue"):	
			self.queue = tf.PaddingFIFOQueue(capacity=10, dtypes=[tf.int32, tf.int32, tf.int32], shapes=[[batch_size, None], [batch_size], [batch_size, None]], shared_name="{}_shared_queue".format(name), name="{}_queue".format(name))
	
		self.batch_size = batch_size
		self.size = self.queue.size()
		self.enqueue_op = self.queue.enqueue([self.inputs, self.inputs_len, self.outputs])
		self.dequeue_op = self.queue.dequeue()
