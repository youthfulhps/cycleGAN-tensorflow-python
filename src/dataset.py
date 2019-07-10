import os
import random
import numpy as np
from datetime import datetime

class Dataset(object):
    def __init__(self, flags):
        self.flags = flags

        self.image_size = (512, 512)
        self.ori_shape = (512, 512)

        self.train_dir = '/Users/Yubyeongho/Desktop/MRI_dataset/train'
        self.test_dir = '/Users/Yubyeongho/Desktop/MRI_dataset/test'

        self.num_train = 0
        self.num_val = 0
        self.num_test = 0

        self._read_data()
        print("num of training images:{}".format(self.num_train))
        print("num of validation images:{}".format(self.num_val))
        print("num of test images:{}".format(self.num_test))

    def _read_data(self):
        if self.flags.is_test:

            self.test_A, self.test_B = utils.get_test_imgs(target_dir=self.test_dir)
            self.test_img_files_A = utils.all_files_under(self.test_dir))
            self.test_img_files_B = utils.all_files_under(self.test_dir))

            self.num_test = self.test_A.shape[0]

        elif not self.flags.is_test:
            self.train_A_files = utils.get_img_path(os.path.join(self.train_dir, 'T1'))
            self.train_B_files = utils.get_img_path(os.path.join(self.train_dir, 'T2'))

            self.num_train = len(self.train_A)

    def train_next_batch(self, batch_size):
        train_indices = np.random.choice(self.num_train, batch_size, replace=True)
        train_A, train_B = utils.get_train_batch(self.train_A_files, sefl.train_B_files, train_indices.astype(np.int32))

        return train_A, train_B
