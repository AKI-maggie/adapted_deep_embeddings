# a improced version which should use TinyImage as source domain
# and use Aptos as target domain

import os
import sys
import random
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import json

target_path = '/srv/scratch/z5141541/data/aptos/aptos'

class TinyImageNet():
    def __init__(self, source_path):
        self.path1 = source_path
        self.path2 = target_path

        self.images1 = []
        self.labels1 = []
        self.images2_train = []
        self.labels2_train = []
        self.images2_test = []
        self.labels2_test = []

    def load_data(self):
        ims1, labels1, ims2_train, labels2_train, ims2_test, labels2_test = self.load(self.path1, self.path2)

        self.images1 = self.process_images(ims1)
        self.labels1 = self.process_labels(labels1)
        self.images2_train = self.process_images(ims2_train)
        self.labels2_train = self.process_labels(labels2_train)
        self.images2_test = self.process_images(ims2_test)
        self.labels2_test = self.process_labels(labels2_test)

        return self.images1, self.images2_train, self.images2_test, self.labels1, self.labels2_train, self.labels2_test

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        return np.array(labels)

    @classmethod
    def extract_from_json(cls, f, root, images, labels):
        # set a limit first
        max_sample_num = 100
        n = 0

        with open(f) as annotations:
            meta = json.load(annotations)
            zipped_li = list(zip(meta['image_names'],meta['image_labels']))
            random.shuffle(zipped_li)
            # pick up 10000 samples in a shuffled order
            for x,y in zipped_li:
                if n > max_sample_num:
                    break
                # transform the image path name
                x = os.path.join(root, x)

                img = imread(x)
                images.append(img)
                labels.append(int(y) + 1)

                n += 1

        return images, labels

    @classmethod
    def load(cls, source_path, target_path):
        class_id = 0
        id_to_label = {}
        validation_annotations = None
        validation_images = {}

        images1 = []
        labels1 = []

        # aptos already have separated training and testing data
        images2_train = []
        labels2_train = []
        images2_test = []
        labels2_test = []

        """
        Load TinyImageNet data
        """
        for root, dirs, files in os.walk(source_path):
            for f in files:
                if f == 'val_annotations.txt':
                    validation_annotations = os.path.join(root, f)
                elif f.endswith('.JPEG'):
                    path = os.path.join(root, f)
                    id = f.split('_')[0]
                    if id == 'val':
                        validation_images[f] = path
                    else:
                        if id not in id_to_label:
                            id_to_label[id] = class_id
                            class_id += 1
                        img = imread(path)
                        if len(img.shape) == 2:
                            img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                        images1.append(img)
                        labels1.append(id_to_label[id])

        with open(validation_annotations) as val_ann:
            for line in val_ann:
                contents = line.split()
                img = imread(validation_images[contents[0]])
                if len(img.shape) == 2:
                    img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                images1.append(img)
                labels1.append(id_to_label[contents[1]])

        """
        Load Aptos data
        """
        for root, dirs, files in os.walk(target_path):
            for f in files:
                f_path = os.path.join(root, f)
                if f == 'base15.json':
                    aptos_train_annotations1 = f_path
                    images2_train, labels2_train = cls.extract_from_json(f_path, "'/srv/scratch/z5141541/data/aptos/", images2_train, labels2_train)
                elif f == 'base19.json':
                    aptos_train_annotations2 = f_path
                    # skip for now
                    continue
                elif f == 'val15.json':
                    aptos_validation_annotations1 = f_path
                    images2_test, labels2_test = cls.extract_from_json(f_path, "'/srv/scratch/z5141541/data/aptos/", images2_test, labels2_test)
                elif f == 'val19.json':
                    aptos_validation_annotations2 = f_path
                    # skip for now
                    continue
                else:
                    continue
        
        return images1, labels1, images2_train, labels2_train, images2_test, labels2_test

    def kntl_data_form(self, k1, n1, k2, n2):
        assert n1 + n2 <= 200
        assert k1 < 550 and k2 < 550
        self.load_data()

        print('Source Domain Full dataset: {0}'.format(len(self.labels1)))

        all_classes = np.unique(self.labels1)
        print('Number of source domain classes: {0}'.format(len(all_classes)))

        print('Target Domain Full dataset: {0}'.format(len(self.labels2_train)))

        all_classes2 = np.unique(self.labels2_train)
        print('Number of target domain classes: {0}'.format(len(all_classes2)))

        # task2_classes = np.sort(np.random.choice(all_classes, n2, replace=False))
        # all_classes = np.delete(all_classes, np.where(np.isin(all_classes, task2_classes)))
        # indices = np.isin(self.labels, task2_classes)
        # self.x_task2, self.y_task2 = self.images[indices], self.labels[indices]
        # self.x_task2, self.y_task2 = self.images2, self.labels2
        # shuffle = np.random.permutation(len(self.y_task2))
        # self.x_task2, self.y_task2 = self.x_task2[shuffle], self.y_task2[shuffle]

        # task1_classes = np.sort(np.random.choice(all_classes, n1, replace=False))
        # indices = np.isin(self.labels, task1_classes)
        # self.x_task1, self.y_task1 = self.images[indices], self.labels[indices]
        self.x_task1, self.y_task1 = self.images1, self.labels1
        shuffle = np.random.permutation(len(self.y_task1))
        self.x_task1, self.y_task1 = self.x_task1[shuffle], self.y_task1[shuffle]

        print('Task 1 Full: {0}'.format(len(self.y_task1)))
        print('Task 2 Full: {0}\n'.format(20))

        # Force class labels to start from 0 and increment upwards by 1
        sorted_class_indices = np.sort(np.unique(self.y_task1))
        zero_based_classes = np.arange(0, len(sorted_class_indices))
        for i in range(len(self.y_task1)):
            self.y_task1[i] = zero_based_classes[sorted_class_indices == self.y_task1[i]]

        self.x_train_task1 = []
        self.y_train_task1 = []
        self.x_valid_task1 = []
        self.y_valid_task1 = []

        for i in zero_based_classes:
            all_indices = np.where(self.y_task1 == i)[0]
            idx = np.random.choice(all_indices, k1, replace=False)
            self.x_train_task1.extend(self.x_task1[idx])
            self.y_train_task1.extend(self.y_task1[idx])
            all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
            self.x_valid_task1.extend(self.x_task1[all_indices])
            self.y_valid_task1.extend(self.y_task1[all_indices])

        self.x_train_task1 = np.array(self.x_train_task1)
        self.y_train_task1 = np.array(self.y_train_task1)
        self.x_valid_task1 = np.array(self.x_valid_task1)
        self.y_valid_task1 = np.array(self.y_valid_task1)

        print('Task 1 training: {0}'.format(len(self.x_train_task1)))
        print('Task 1 validation: {0}'.format(len(self.x_valid_task1)))

        # Force class labels to start from 0 and increment upwards by 1
        # sorted_class_indices = np.sort(np.unique(self.y_task2))
        # zero_based_classes = np.arange(0, len(sorted_class_indices))
        # for i in range(len(self.y_task2)):
        #     self.y_task2[i] = zero_based_classes[sorted_class_indices == self.y_task2[i]]

        self.x_train_task2 = self.images2_train
        self.y_train_task2 = self.labels2_train
        # for i in zero_based_classes:
        #     idx = np.random.choice(np.where(self.y_task2 == i)[0], k2, replace=False)
        #     self.x_train_task2.extend(self.x_task2[idx])
        #     self.y_train_task2.extend(self.y_task2[idx])
        #     self.x_task2 = np.delete(self.x_task2, idx, axis=0)
        #     self.y_task2 = np.delete(self.y_task2, idx, axis=0)

        self.x_train_task2 = np.array(self.x_train_task2)
        self.y_train_task2 = np.array(self.y_train_task2)

        # k_test = 550 - k2

        self.x_test_task2 = self.images2_test
        self.y_test_task2 = self.labels2_test
        # for i in zero_based_classes:
        #     idx = np.random.choice(np.where(self.y_task2 == i)[0], k_test, replace=False)
        #     self.x_test_task2.extend(self.x_task2[idx])
        #     self.y_test_task2.extend(self.y_task2[idx])

        self.x_test_task2 = np.array(self.x_test_task2)
        self.y_test_task2 = np.array(self.y_test_task2)

        print('k = {0}, n = {1}'.format(k2, n2))
        print('Task 2 training: {0}'.format(len(self.x_train_task2)))
        print('Task 2 test: {0}\n'.format(len(self.x_test_task2)))

        return (self.x_train_task1, self.y_train_task1), (self.x_valid_task1, self.y_valid_task1), (self.x_train_task2, self.y_train_task2), (self.x_test_task2, self.y_test_task2)
