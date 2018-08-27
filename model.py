import numpy as np
import tensorflow as tf 

class ANN:
	def __init__(self, shape):
		W, H = shape[-2], shape[-1]
		print(shape)
		print(W,H)
		tf.reset_default_graph()
		self.x = tf.placeholder(tf.float32,[None,W,H])
		self.y = tf.placeholder(tf.float32,[None,10])

		W1 = tf.Variable(tf.random_normal([W * H,64],stddev=0.1))
		b1 = tf.Variable(tf.random_normal([64],stddev=0.1))

		W2 = tf.Variable(tf.random_normal([64,10],stddev=0.1))
		b2 = tf.Variable(tf.random_normal([10],stddev=0.1))

		x_ = tf.reshape(self.x,[-1,W * H])
		x1 = tf.matmul(x_,W1)+b1
		x1_a = tf.nn.tanh(x1)
		x2 = tf.matmul(x1_a,W2)+b2
		self.output = tf.nn.softmax(x2)
		self.loss = tf.losses.softmax_cross_entropy(self.y,self.output)
		self.optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def train(self, tr_data, tr_labels, num_iter=5000):
		for i in range(num_iter):
		    indices = np.random.randint(len(tr_data),size=[64])
		    b_data = tr_data[indices]
		    b_labels = tr_labels[indices]
		    
		    self.sess.run(self.optim,feed_dict={self.x:b_data,self.y:b_labels})
		    
		    if i % (num_iter/10) == 0:
		        print("LOSS : " + str(self.sess.run(self.loss,feed_dict={self.x:tr_data,self.y:tr_labels})))
		        preds = self.sess.run(self.output,feed_dict={self.x:tr_data})
		        print("ACC : " + str(np.mean(np.equal(np.argmax(preds,axis=1),np.argmax(tr_labels,axis=1)))))




