# import tensorflow as tf
# import numpy as np
# # import matplotlib.pyplot as plt
#
# from tensorflow.examples.tutorials.mnist import input_data
#
# tf.set_random_seed(777)  # reproducibility
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# x = tf.placeholder(tf.float32,[None,784])
# x_img = tf.reshape(x,[-1,28,28,1])
# y = tf.placeholder(tf.float32,[None,10])
# keep_prob = tf.placeholder(tf.float32)
# conv1 = tf.layers.conv2d(inputs=x_img,filters=32,kernel_size=[3,3],padding='SAME',activation=tf.nn.relu)
# pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],padding='SAME',strides=2)
# dropout1 = tf.layers.dropout(inputs=pool1,rate=0.7,training=True)
#
# conv2 = tf.layers.conv2d(inputs=dropout1,filters=64,kernel_size=[3,3],padding='SAME',activation=tf.nn.relu)
# pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],padding='SAME',strides=2)
# dropout2 = tf.layers.dropout(inputs=pool2,rate=0.7,training=True)
#
# conv3 = tf.layers.conv2d(inputs=dropout1,filters=64,kernel_size=[3,3],padding='SAME',activation=tf.nn.relu)
# pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],padding='SAME',strides=2)
# dropout3 = tf.layers.dropout(inputs=pool3,rate=0.7,training=True)
#
# flat = tf.reshape(dropout3,[-1,64*7*7])
# dense4 = tf.layers.dense(inputs=flat,units=625,activation=tf.nn.relu)
# dropout4 = tf.layers.dropout(inputs=dense4,rate=0.5,training=True)
#
# logits = tf.layers.dense(inputs=dropout4,units=10)
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
# opti = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(cost)
#
#
# corr_predict = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(corr_predict,tf.float32))
#
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# batch_size = 100
# training_epoch = 15
#
# for epoch in range(training_epoch):
#     avg_cost = 0
#     total_batch = int(mnist.test.num_examples/batch_size)
#
#     for i in range(total_batch):
#         xs,ys = mnist.train.next_batch(batch_size)
#         c,_ = sess.run([cost,opti],feed_dict={x:xs,y:ys,keep_prob:0.7})
#         avg_cost = c/total_batch
#
#     print 'epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost)
# print 'accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1})





################

# import tensorflow as tf
# # import matplotlib.pyplot as plt
#
# from tensorflow.examples.tutorials.mnist import input_data
#
# tf.set_random_seed(777)  # reproducibility
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# # Check out https://www.tensorflow.org/get_started/mnist/beginners for
# # more information about the mnist dataset
#
# # hyper parameters
# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
#
#
# class Model:
#
#     def __init__(self, sess, name):
#         self.sess = sess
#         self.name = name
#         self._build_net()
#
#     def _build_net(self):
#         with tf.variable_scope(self.name):
#             # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
#             # for testing
#             self.training = tf.placeholder(tf.bool)
#
#             # input place holders
#             self.X = tf.placeholder(tf.float32, [None, 784])
#
#             # img 28x28x1 (black/white), Input Layer
#             X_img = tf.reshape(self.X, [-1, 28, 28, 1])
#             self.Y = tf.placeholder(tf.float32, [None, 10])
#
#             # Convolutional Layer #1
#             conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
#                                      padding="SAME", activation=tf.nn.relu)
#             # Pooling Layer #1
#             pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
#                                             padding="SAME", strides=2)
#             dropout1 = tf.layers.dropout(inputs=pool1,
#                                          rate=0.7, training=self.training)
#
#             # Convolutional Layer #2 and Pooling Layer #2
#             conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
#                                      padding="SAME", activation=tf.nn.relu)
#             pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
#                                             padding="SAME", strides=2)
#             dropout2 = tf.layers.dropout(inputs=pool2,
#                                          rate=0.7, training=self.training)
#
#             # Convolutional Layer #2 and Pooling Layer #2
#             conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
#                                      padding="same", activation=tf.nn.relu)
#             pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
#                                             padding="same", strides=2)
#             dropout3 = tf.layers.dropout(inputs=pool3,
#                                          rate=0.7, training=self.training)
#
#             # Dense Layer with Relu
#             flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
#             dense4 = tf.layers.dense(inputs=flat,
#                                      units=625, activation=tf.nn.relu)
#             dropout4 = tf.layers.dropout(inputs=dense4,
#                                          rate=0.5, training=self.training)
#
#             # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
#             self.logits = tf.layers.dense(inputs=dropout4, units=10)
#
#         # define cost/loss & optimizer
#         self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#             logits=self.logits, labels=self.Y))
#         self.optimizer = tf.train.AdamOptimizer(
#             learning_rate=learning_rate).minimize(self.cost)
#
#         correct_prediction = tf.equal(
#             tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
#         self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     def predict(self, x_test, training=False):
#         return self.sess.run(self.logits,
#                              feed_dict={self.X: x_test, self.training: training})
#
#     def get_accuracy(self, x_test, y_test, training=False):
#         return self.sess.run(self.accuracy,
#                              feed_dict={self.X: x_test,
#                                         self.Y: y_test, self.training: training})
#
#     def train(self, x_data, y_data, training=True):
#         return self.sess.run([self.cost, self.optimizer], feed_dict={
#             self.X: x_data, self.Y: y_data, self.training: training})
#
# # initialize
# sess = tf.Session()
# m1 = Model(sess, "m1")
#
# sess.run(tf.global_variables_initializer())
#
# print('Learning Started!')
#
# # train my model
# for epoch in range(training_epochs):
#     avg_cost = 0
#     total_batch = int(mnist.train.num_examples / batch_size)
#
#     for i in range(total_batch):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         c, _ = m1.train(batch_xs, batch_ys)
#         avg_cost += c / total_batch
#
#     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
#
# print('Learning Finished!')
#
# # Test model and check accuracy
# print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
#

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.01

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2], name='x-input')
Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)


with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# cost/loss function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as scope:
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
    writer.add_graph(sess.graph)  # Show the graph

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


'''
Hypothesis:  [[  6.13103184e-05]
 [  9.99936938e-01]
 [  9.99950767e-01]
 [  5.97514772e-05]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0
'''