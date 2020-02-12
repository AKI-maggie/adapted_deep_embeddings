# a improced version which should use TinyImage as source domain
# and use Aptos as target domain

import os
import sys
import random
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import json
import cv2

target_path = '/srv/scratch/z5141541/data/aptos/aptos/'

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
        # leave img_ids first
        self.images2_train = ims2_train
        self.labels2_train = self.process_labels(labels2_train)
        self.images2_test = ims2_test
        self.labels2_test = self.process_labels(labels2_test)

        return self.images1, self.images2_train, self.images2_test, self.labels1, self.labels2_train, self.labels2_test

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        return np.array(labels)

    # extract image ids and labels from Aptos dir
    @classmethod
    def extract_from_json(cls, f, root, images, labels):       
        # set up class subset
        # all_class = range(5)
        zipped_li = [] 

        image_ids = []
        labels = []

        # get full pairs
        with open(f) as annotations:
            meta = json.load(annotations)
            zipped_li = list(zip(meta['image_names'],meta['image_labels']))
            random.shuffle(zipped_li)

            # pick up 10000 samples in a shuffled order
            for x,y in zipped_li:
                # if n > max_sample_num:
                    # break
                # transform the image path name
                x = os.path.join(root, x)
                image_ids.append(x)
                # img = imread(x)
                # resize the image to the same size with tiny image
                # images.append(cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC))
                labels.append(int(y) + 1)
        
        return image_ids, labels

    @classmethod
    def load_aptos(cls, selected_img_ids):
        # check the first one
        print(selected_img_ids[0])

        imgs = []
        for each in selected_img_ids:
            img = imread(each)
            imgs.append(cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC))        
        return imgs

    # extract required subsets and load aptos images 
    @classmethod
    def subset_aptos(cls, img_ids, labels, k, n):
        x = []
        y = []
        print(labels[:5])
        # support 2-label of 5-label
        classes = []
        if n == 2:
            classes = [1, 5]
        else:
            classes = range(1,6)

        # choose image subsets and load images according to image ids
        img_ids = np.array(img_ids)
        labels = np.array(labels)

        indices = np.isin(labels, classes)
        task_img_ids, task_labels = img_ids[indices], labels[indices]
        shuffle = np.random.permutation(len(task_labels))
        task_img_ids, task_labels = task_img_ids[shuffle], task_labels[shuffle]

        print(task_labels.shape)
        print(task_labels[:5])
        # generate k-shot n-class dataset
        for i in classes:
            print("Choosing "+str(i))
            all_indices = np.where(task_labels == i)[0]
            print(all_indices.shape)
            print(k)
            idx = np.random.choice(all_indices, k, replace=False)

            x.extend(cls.load_aptos(task_img_ids[idx]))
            y.extend(task_labels[idx])

        x = np.array(x)
        y = np.array(y)

        print('Task 1 training: {0}'.format(len(x)))
        print('Task 1 validation: {0}'.format(len(y)))

        return x, y
        

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
                    images2_train, labels2_train = cls.extract_from_json(f_path, "/srv/scratch/z5141541/data/aptos/", images2_train, labels2_train)
                elif f == 'base19.json':
                    aptos_train_annotations2 = f_path
                    # skip for now
                    continue
                elif f == 'val15.json':
                    aptos_validation_annotations1 = f_path
                    images2_test, labels2_test = cls.extract_from_json(f_path, "/srv/scratch/z5141541/data/aptos/", images2_test, labels2_test)
                elif f == 'val19.json':
                    aptos_validation_annotations2 = f_path
                    # skip for now
                    continue
                else:
                    continue
        
        return images1, labels1, images2_train, labels2_train, images2_test, labels2_test

    def kntl_data_form(self, k1, n1, k2, n2):
        # assert n1 + n2 <= 200
        assert k1 < 550 # and k2 < 550
        self.load_data()

        all_classes = np.unique(self.labels1)

        """
        Generate source domain training sets (TinyImageNet)
        """
        task1_classes = np.sort(np.random.choice(all_classes, n1, replace=False))
        indices = np.isin(self.labels1, task1_classes)
        self.x_task1, self.y_task1 = self.images1[indices], self.labels1[indices]
        shuffle = np.random.permutation(len(self.y_task1))
        self.x_task1, self.y_task1 = self.x_task1[shuffle], self.y_task1[shuffle]

        print('Task 1 Full: {0}'.format(len(self.y_task1)))

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

        """
        Generate target domain training sets (Aptos)
        """
        self.x_train_task2, self.y_train_task2 = self.subset_aptos(self.images2_train, self.labels2_train, k2, n2)
        self.x_test_task2, self.y_test_task2 = self.subset_aptos(self.images2_test, self.labels2_test, k2, n2)

        self.x_train_task2 = self.process_images(self.x_train_task2)
        self.x_test_task2 = self.process_images(self.x_test_task2)

        print(self.x_train_task2.shape)
        print(self.y_train_task2.shape)
        print(self.x_test_task2.shape)
        print(self.y_test_task2.shape)

        print('k = {0}, n = {1}'.format(k2, n2))
        print('Task 2 training: {0}'.format(len(self.x_train_task2)))
        print('Task 2 test: {0}\n'.format(len(self.x_test_task2)))

        return (self.x_train_task1, self.y_train_task1), (self.x_valid_task1, self.y_valid_task1), (self.x_train_task2, self.y_train_task2), (self.x_test_task2, self.y_test_task2)
