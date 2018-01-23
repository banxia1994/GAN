# from __future__ import print_function
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
# from six.moves import xrange
# import tensorflow.contrib.slim as slim
# import os
# import tensorflow as tf
# import numpy as np
# import tensorflow.contrib.layers as ly
# #from load_svhn import load_svhn
# from tensorflow.examples.tutorials.mnist import input_data
# from functools import partial
#
# def lrelu(x, leak=0.3, name="lrelu"):
#     with tf.variable_scope(name):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)
#
# batch_size = 64
# z_dim = 128
# learning_rate_ger = 5e-5
# learning_rate_dis = 5e-5
# device = '/gpu:1'
# # img size
# s = 32
# # update Citers times of critic in one iter(unless i < 25 or i % 500 == 0, i is iterstep)
# Citers = 5
# # the upper bound and lower bound of parameters in critic
# clamp_lower = -0.01
# clamp_upper = 0.01
# # whether to use mlp or dcgan stucture
# is_mlp = False
# # whether to use adam for parameter update, if the flag is set False, use tf.train.RMSPropOptimizer
# # as recommended in paper
# is_adam = False
# # whether to use SVHN or MNIST, set false and MNIST is used
# is_svhn = False
# channel = 3 if is_svhn is True else 1
# # 'gp' for gp WGAN and 'regular' for vanilla
# mode = 'gp'
# # if 'gp' is chosen the corresponding lambda must be filled
# lam = 10.
# s2, s4, s8, s16 =\
#     int(s / 2), int(s / 4), int(s / 8), int(s / 16)
# # hidden layer size if mlp is chosen, ignore if otherwise
# ngf = 64
# ndf = 64
# # directory to store log, including loss and grad_norm of generator and critic
# log_dir = './log_wgan'
# ckpt_dir = './ckpt_wgan'
# if not os.path.exists(ckpt_dir):
#     os.makedirs(ckpt_dir)
# # max iter step, note the one step indicates that a Citers updates of critic and one update of generator
# max_iter_step = 20000
#
#
# def generator_conv(z):
#     train = ly.fully_connected(
#         z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#     train = tf.reshape(train, (-1, 4, 4, 512))
#     train = ly.conv2d_transpose(train, 256, 3, stride=2,
#                                 activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
#     train = ly.conv2d_transpose(train, 128, 3, stride=2,
#                                 activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
#     train = ly.conv2d_transpose(train, 64, 3, stride=2,
#                                 activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
#     train = ly.conv2d_transpose(train, channel, 3, stride=1,
#                                 activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
#     print(train.name)
#     return train
#
# def generator_mlp(z):
#     train = ly.fully_connected(
#         z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#     train = ly.fully_connected(
#         train, ngf, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#     train = ly.fully_connected(
#         train, ngf, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#     train = ly.fully_connected(
#         train, s*s*channel, activation_fn=tf.nn.tanh, normalizer_fn=ly.batch_norm)
#     train = tf.reshape(train, tf.stack([batch_size, s, s, channel]))
#     return train
#
# def critic_conv(img, reuse=False):
#     with tf.variable_scope('critic') as scope:
#         if reuse:
#             scope.reuse_variables()
#         size = 64
#         img = ly.conv2d(img, num_outputs=size, kernel_size=3,
#                         stride=2, activation_fn=lrelu)
#         img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3,
#                         stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#         img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3,
#                         stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#         img = ly.conv2d(img, num_outputs=size * 8, kernel_size=3,
#                         stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#         logit = ly.fully_connected(tf.reshape(
#             img, [batch_size, -1]), 1, activation_fn=None)
#     return logit
#
# def critic_mlp(img, reuse=False):
#     with tf.variable_scope('critic') as scope:
#         if reuse:
#             scope.reuse_variables()
#         size = 64
#         img = ly.fully_connected(tf.reshape(
#             img, [batch_size, -1]), ngf, activation_fn=tf.nn.relu)
#         img = ly.fully_connected(img, ngf,
#             activation_fn=tf.nn.relu)
#         img = ly.fully_connected(img, ngf,
#             activation_fn=tf.nn.relu)
#         logit = ly.fully_connected(img, 1, activation_fn=None)
#     return logit
#
# def build_graph():
# #     z = tf.placeholder(tf.float32, shape=(batch_size, z_dim))
#     noise_dist = tf.contrib.distributions.Normal(0., 1.)
#     z = noise_dist.sample((batch_size, z_dim))
#     generator = generator_mlp if is_mlp else generator_conv
#     critic = critic_mlp if is_mlp else critic_conv
#     with tf.variable_scope('generator'):
#         train = generator(z)
#     real_data = tf.placeholder(
#         dtype=tf.float32, shape=(batch_size, 32, 32, channel))
#     true_logit = critic(real_data)
#     fake_logit = critic(train, reuse=True)
#     c_loss = tf.reduce_mean(fake_logit - true_logit)
#     if mode is 'gp':
#         alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
#         alpha = alpha_dist.sample((batch_size, 1, 1, 1))
#         interpolated = real_data + alpha*(train-real_data)
#         inte_logit = critic(interpolated, reuse=True)
#         gradients = tf.gradients(inte_logit, [interpolated,])[0]
#         grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
#         gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
#         gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
#         grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
#         c_loss += lam*gradient_penalty
#     g_loss = tf.reduce_mean(-fake_logit)
#     g_loss_sum = tf.summary.scalar("g_loss", g_loss)
#     c_loss_sum = tf.summary.scalar("c_loss", c_loss)
#     img_sum = tf.summary.image("img", train, max_outputs=10)
#     theta_g = tf.get_collection(
#         tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
#     theta_c = tf.get_collection(
#         tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
#     counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
#     opt_g = ly.optimize_loss(loss=g_loss, learning_rate=learning_rate_ger,
#                     optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer,
#                     variables=theta_g, global_step=counter_g,
#                     summaries = ['gradient_norm'])
#     counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
#     opt_c = ly.optimize_loss(loss=c_loss, learning_rate=learning_rate_dis,
#                     optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer,
#                     variables=theta_c, global_step=counter_c,
#                     summaries = ['gradient_norm'])
#     if mode is 'regular':
#         clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]
#         # merge the clip operations on critic variables
#         with tf.control_dependencies([opt_c]):
#             opt_c = tf.tuple(clipped_var_c)
#     if not mode in ['gp', 'regular']:
#         raise(NotImplementedError('Only two modes'))
#     return opt_g, opt_c, real_data
#
# def main():
#     if is_svhn is True:
#         dataset = load_svhn()
#     else:
#         dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
#     with tf.device(device):
#         opt_g, opt_c, real_data = build_graph()
#     merged_all = tf.summary.merge_all()
#     saver = tf.train.Saver()
#     config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#     config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.8
#     def next_feed_dict():
#         train_img = dataset.train.next_batch(batch_size)[0]
#         train_img = 2*train_img-1
#         if is_svhn is not True:
#             train_img = np.reshape(train_img, (-1, 28, 28))
#             npad = ((0, 0), (2, 2), (2, 2))
#             train_img = np.pad(train_img, pad_width=npad,
#                                mode='constant', constant_values=-1)
#             train_img = np.expand_dims(train_img, -1)
#         feed_dict = {real_data: train_img}
#         return feed_dict
#     with tf.Session(config=config) as sess:
#         sess.run(tf.global_variables_initializer())
#         summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
#         for i in range(max_iter_step):
#             if i < 25 or i % 500 == 0:
#                 citers = 100
#             else:
#                 citers = Citers
#             for j in range(citers):
#                 feed_dict = next_feed_dict()
#                 if i % 100 == 99 and j == 0:
#                     run_options = tf.RunOptions(
#                         trace_level=tf.RunOptions.FULL_TRACE)
#                     run_metadata = tf.RunMetadata()
#                     _, merged = sess.run([opt_c, merged_all], feed_dict=feed_dict,
#                                          options=run_options, run_metadata=run_metadata)
#                     summary_writer.add_summary(merged, i)
#                     summary_writer.add_run_metadata(
#                         run_metadata, 'critic_metadata {}'.format(i), i)
#                 else:
#                     sess.run(opt_c, feed_dict=feed_dict)
#             feed_dict = next_feed_dict()
#             if i % 100 == 99:
#                 _, merged = sess.run([opt_g, merged_all], feed_dict=feed_dict,
#                      options=run_options, run_metadata=run_metadata)
#                 summary_writer.add_summary(merged, i)
#                 summary_writer.add_run_metadata(
#                     run_metadata, 'generator_metadata {}'.format(i), i)
#             else:
#                 sess.run(opt_g, feed_dict=feed_dict)
#             if i % 1000 == 999:
#                 saver.save(sess, os.path.join(
#                     ckpt_dir, "model.ckpt"), global_step=i)
#
# main()

import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc

from WGANvisualize import *


class WassersteinGAN(object):
    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
                .minimize(self.d_loss_reg, var_list=self.d_net.vars)
            self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
                .minimize(self.g_loss_reg, var_list=self.g_net.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=64, num_batches=1000000):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            if t % 500 == 0 or t < 25:
                 d_iters = 100

            for _ in range(0, d_iters):
                 bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)
                self.sess.run(self.d_clip)
                self.sess.run(self.d_rmsprop, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim)
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.x: bx})

            if t % 100 == 0:
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz, self.x: bx}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss - g_loss, g_loss))

            if t % 100 == 0:
                bz = self.z_sampler(batch_size, self.z_dim)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)
                fig = plt.figure(self.data + '.' + self.model)
                grid_show(fig, bx, xs.shape)
                fig.savefig('logs/{}/{}.pdf'.format(self.data, t/100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, xs, zs, args.data, args.model)
    wgan.train()