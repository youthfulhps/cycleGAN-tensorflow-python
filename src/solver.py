import os
import time
import collections
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from PIL import Image

from dataset import Dataset
import layers
import utils
from model import CycleGAN

class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session()

        self.flags = flags
        self.dataset = Dataset(self.flags)
        self.model = CGAN(self.sess, self.flags, self.dataset.image_size)

        #self.score = 0.
        self._make_folders()

    def _make_folders(self):
        self.model_out_dir = "{}/model_{}_{}".format(self.flags.output_dir, self.flags.lambda_cycle ,self.flags.lambda_identity)

        if not os.path.isdir(self.model_out_dir):
            os.makedirs(self.model_out_dir)

        if self.flags.is_test:
            self.img_out_dir = "{}/seg_result_gan*{}+seg*{}/{}".format(self.flags.output_dir, self.flags.lambda_cycle ,self.flags.lambda_identity, self.flags.fn)
            #self.auc_out_dir = "{}/score_gan*{}+seg*{}".format(self.flags.output_dir, self.flags.lambda2 ,self.flags.lambda1)

            if not os.path.isdir(self.img_out_dir):
                os.makedirs(self.img_out_dir)
            '''
            if not os.path.isdir(self.auc_out_dir):
                os.makedirs(self.auc_out_dir)
            '''
    def train(self):
        for iter_time in range(0, self.flags.iters+1, self.flags.train_interval):
            self.sample(iter_time)

            for iter_ in range(1, self.flags.train_interval+1):
                real_A, real_B = self.dataset.train_next_batch(batch_size=self.flags.batch_size)
                g_loss, d_loss = self.model.train(real_A, real_B, iter_time)

                self.print_info(iter_time + iter_, 'g_loss', g_loss)
                self.print_info(iter_time + iter_, 'd_loss', d_loss)

            if np.mod(iter_time, self.flags.save_freq)==0:
                self.save_model(iter_time)

    def test(self):
        if self.load_model():
            print('[*] Load Success!\n')
            #self.eval(phase='test')
        else:
            print('[!] Load Failed!\n')

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq)==0:
            print('Generating Validation Data A from B...')
            real_B, samples_A = [], []
            for i in range(self.dataset.num_test):
                real_B.append(self.dataset.test_A[i])

            for j in range(len(real_B)):
                sample = self.model.test(np.expand_dims(real_B[j],axis=0), direction='B2A')
                samples_A.append(sample)

                self.plot(samples, save_file="{}/".format(self.sample_out_dir), phase='test')

        if np.mod(iter_time, self.flags.sample_freq)==0:
            print('Generating Validation Data B from A...')
            real_A, samples_B = [], []
            for i in range(self.dataset.num_test):
                real_A.append(self.dataset.test_B[i])

            for j in range(self.dataset.num_test):
                sample = self.model.test(np.expand_dims(real_A[j],axis=0), direction='A2B')
                samples_B.append(sample)

                self.plot(samples, save_file="{}/".format(self.sample_out_dir), phase='test')

    def print_info(self, iter_time, name, loss):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([(name, loss),
                                                  ('train_interval', np.float32(self.flags.train_interval)),
                                                  ('gpu_index', self.flags.gpu_index)])
            utils.print_metrics(iter_time, ord_output)

    #def eval()
    #def measure()

    def save_model(self, iter_time):

        model_name = "iter_{}_{}_{}".format(iter_time, self.flags.lambda_cycle, self.flags.lambda_identity)
        self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name))
        print('====================================================')
        print('                    Model saved!                    ')
        print('====================================================')



    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.flags.model_dir)
        all_models = ckpt.all_model_checkpoint_paths
        if ckpt and all_models:
            ckpt_name = os.path.basename(all_models[self.flags.mn])
            self.saver.restore(self.sess, os.path.join(self.flags.model_dir, ckpt_name))

            #self.score = self.sess.run(self.model.score)
            print('====================================================')
            print('                     Model saved!                   ')
            print(' Score: {:.3}'.format(self.score))
            print('====================================================')

            return True
        else:
            return False
