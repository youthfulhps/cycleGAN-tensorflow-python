import tensorflow as tf
from layers import *
import os
import random
import numpy as np

def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))

def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

def least_loss(y, y_hat):
    return tf.reduce_mean(tf.square(y - y_hat))

def pixel_loss(y, y_hat):
    return tf.reduce_mean(tf.abs(y-y_hat))

class CycleGAN(object):
    def __init__(self, sess, flags, image_size):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size  #(256,256,3)

        self._gen_train_ops, self._dis_train_ops = [], []
        self.gen_c, self.dis_c = 64, 64

        self._build_net()
        self._init_assign_op()

        print('Initialized CycleGAN SUCCESS!\n')

    def _build_net(self):
        self.real_A = tf.placeholder(tf.float32, shape=self.input_size, name='real_A')
        sefl.real_B = tf.placeholder(tf.float32, shape=self.input_size, name='real_B')

        self.fake_A = tf.placeholder(tf.float32, shape=self.input_size, name='fake_A')
        self.fake_B = tf.placeholder(tf.float32, shape=self.input_size, name='fake_B')

        self.test_A = tf.placeholder(tf.float32, shape=self.input_size, name='test_A')
        self.test_B = tf.placeholder(tf.float32, shape=self.input_szie, name='test_B')

        self.generation_B = self.generator(x = self.real_A, reuse=False, scope_name='generator_A2B')
        self.cycle_A = self.generator(x = self.generation_B, reuse=False, scope_name='generator_B2A')

        self.generation_A = self.generator(x = self.real_B, reuse=True, scope_name='generator_B2A')
        self.cycle_B = self.generator(x = self.generation_A, reuse=True, scope_name='generator_A2B')

        self.identity_A = self.generator(x = self.real_A, reuse=True, scope_name='generator_B2A')
        self.identity_B = self.generator(x = self.real_B, reuse=True, scope_name='generator_A2B')

        self.discrimination_A_fake = self.discriminator(x = self.generation_A, reuse=False, scope_name='discriminator_A')
        self.discrimination_B_fake = self.discriminator(x = self.generation_B, reuse=False, scope_name='discriminator_B')

        self.cycle_loss = l1_loss(y = self.real_A, y_hat=self.cycle_A) + l1_loss(y=self.real_B, y_hat=self.cycle_B)

        self.identity_loss = l1_loss(y=self.real_A, y_hat=self.identity_A) + l1_loss(y=self.real_B, y_hat=self.identity_B)

        self.lambda_cycle = tf.placeholder(tf.float32, None, name='lambda_cycle')
        self.lambda_identity = tf.placeholder(tf.float32, None, name='lambda_identity')

        self.generator_loss_A2B = l2_loss(y=tf.ones_like(self.discrimination_B_fake),y_hat=self.discrimination_B_fake)
        self.generator_loss_B2A = l2_loss(y=tf.ones_like(self.discrimination_A_fake),y_hat=self.discrimination_A_fake)

        self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + (self.lambda_cycle * self.cycle_loss) + (self.lambda_identity * self.identity_loss)

        self.discrimination_A_real = self.discriminator(x = self.real_A, reuse=True, scope_name='discriminator_A')
        self.discrimination_B_real = self.discriminator(x = self.real_B, reuse=True, scope_name='discriminator_B')

        self.discrimination_A_fake = self.discriminator(x = self.fake_A, reuse=True, scope_name='discriminator_A')
        self.discrimination_B_fake = self.discriminator(x = self.fake_B, reuse=True, scope_name='discriminator_B')

        self.discriminator_loss_A_real = l2_loss(y=tf.ones_like(self.discrimination_A_real),y_hat = self.discrimination_A_real)
        self.discriminator_loss_A_fake = l2_loss(y=tf.zeros_like(self.discrimination_A_fake),y_hat = self.discrimination_A_fake)
        self.discriminator_loss_A = (self.discriminator_loss_A_real + self.discriminator_loss_A_fake) /2

        self.discriminator_loss_B_real = l2_loss(y=tf.ones_like(self.discrimination_B_real),y_hat = self.discrimination_B_real)
        self.discriminator_loss_B_fake = l2_loss(y=tf.zeros_like(self.discrimination_B_fake),y_hat = self.discrimination_B_fake)
        self.discriminator_loss_B = (self.discriminator_loss_B_real + self.discriminator_loss_B_fake) /2

        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B

        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator_' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator_' in var.name]

        self.generation_B_test = self.generator(x = self.test_A, reuse=True, scope_name='generator_A2B')
        self.generation_A_test = self.generator(x = self.test_B, reuse=True, scope_name='generator_B2A')

        self.discriminator_op = tf.train.AdamOptimizer(learning_rate = self.flags.learning_rate, beta1=self.flags.beta1).minimize(self.discriminator_loss, var_list=self.discriminator_vars)
        self.dis_ops = [self.discriminator_op] + self._dis_train_ops
        self.dis_optim = tf.group(*self.dis_ops)

        self.generator_op = tf.train.AdamOptimizer(learning_rate = self.flags.learning_rate, beta1=self.flags.beta1).minimize(self.generator_loss, var_list=self.generator_vars)
        self.gen_ops = [self.generator_op] + self._gen_train_ops
        self.gen_optim = tf.group(*self.gen_ops)

    def _init_assign_op(self):

        self.psnr_placeholder = tf.placeholder(tf.float32, name='psnr_placeholder')
        self.ssim_placeholder = tf.placeholder(tf.float32, name='ssim_placeholder')
        self.score_placeholder = tf.placeholder(tf.float32, name='score_best_placeholder')

        psnr = tf.Variable(0., trainable=False, dtype=tf.float32, name='psnr')
        ssim = tf.Variable(0., trainable=False, dtype=tf.float32, name='ssim')
        self.score = tf.Variable(0., trainable=False, dtype=tf.float32, name='score_best')

        self.score_assign_op = self.score.assign(self.score_placeholder)
        psnr_assign_op = psnr.assign(self.psnr_placeholder)
        ssim_assign_op = ssim.assign(self.ssim_placeholder)

        self.measure_assign_op = tf.group(psnr_assign_op, ssim_assign_op)
        self.model_out_dir = "{}/model_gan*{}+seg*{}".format(self.flags.output_dir, self.flags.lambda2 ,self.flags.lambda1)
        # for tensorboard
        if not self.flags.is_test:
            self.writer = tf.summary.FileWriter("{}/logs/gan*{}+seg*{}".format(self.flags.output_dir, self.flags.lambda2, self.flags.lambda1))


        psnr_summ = tf.summary.scalar("psnr_summary", psnr)
        ssim_summ = tf.summary.scalar("ssim_summary", ssim)
        score_summ = tf.summary.scalar("score_summary", self.score)

        self.g_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
        self.d_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)

        self.measure_summary = tf.summary.merge([psnr_summ, ssim_summ, score_summ])


    def generator(self, x, reuse=False, scope_name='generator_'):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            with tf.variable_scope(name):
        # image is 512, 512 x input_c_dim


        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x


        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred

    def discriminator(image, options, reuse=False, name="discriminator"):

        with tf.variable_scope(name):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
            h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
            return h4
