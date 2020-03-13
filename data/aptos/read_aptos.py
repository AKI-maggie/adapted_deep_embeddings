import os
import sys
import random
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import json
import cv2

class Aptos():
    def __init__(self, path):
        self.path = path

        self.imgs = []
        self.img_ids = []
        self.labels = []

        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []

    def read_data(self, domain_option):
        imgs = []
        img_ids = []
        labels = []

        for root, dirs, files in os.walk(self.path):
            for f in files:
                fpath = os.path.join(root, f)
                if f == "base15.json":
                    t1, t2, t3 = self.register_imgs(domain_option, fpath, 300)
                    imgs.extend(t1)
                    img_ids.extend(t2)
                    labels.extend(t3)
                # if f == "novel15.json":
                    # t1, t2, t3 = self.register_imgs(domain_option, fpath, 150)
                    # imgs.extend(t1)
                    # img_ids.extend(t2)
                    # labels.extend(t3)
        
        print("Totally loaded {0} imgs.".format(len(imgs)))

        imgs = self.process_images(imgs)
        img_ids = np.array(img_ids)
        labels = self.process_labels(labels)

        return imgs, img_ids, labels

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        return np.array(labels)

    # domain_option = 
    #   0 - images of all classes 
    #   1 - only images with labels of 0 and 4
    #   2 - only images with labels of 0, 2 and 4
    def register_imgs(self, domain_option, fpath, limit):
        imgs = []
        img_ids = []
        labels = []
        zipped_li = []
        load_imgs_count = [0, 0, 0, 0, 0]

        with open(fpath) as ann:
            meta = json.load(ann)
        
        zipped_li = list(zip(meta['image_names'], meta['image_labels']))
        random.shuffle(zipped_li)

        r_labels = [0, 1, 2, 3, 4]
        if domain_option == 1:
            ranges = [0, 4]
        elif domain_option == 2:
            ranges = [0, 2, 4]

        for x, y in zipped_li:
            num_y = int(y)
            if num_y in r_labels and load_imgs_count[num_y] < limit:
                img = imread(x)
                # print("Image shape: {0}".format(img.shape))

                imgs.append(cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC))
                img_ids.append(x)
                labels.append(num_y)
                load_imgs_count[num_y] += 1
        
        print("Path {0} has loaded: ".format(fpath))
        for i in range(5):
            print("Class {0}: {1} imgs".format(i, load_imgs_count[i]))
        
        return imgs, img_ids, labels

    def load_data(self, domain_option):
        self.imgs, self.img_ids, self.labels = self.read_data(domain_option)


    def kntl_data_form(self, k, option):
        self.load_data(option)

        self.generate_domain(k)
        return (self.x_train, self.y_train), (self.x_valid, self.y_valid), \
               (np.array([]), np.array([])), (np.array([]), np.array([]))

    def generate_domain(self, k):
        all_classes = np.unique(self.labels)

        shuffle = np.random.permutation(self.labels.shape[0])

        x_task, y_task, z_task = self.imgs[shuffle], self.labels[shuffle], self.img_ids[shuffle]

        sorted_class_index = np.sort(all_classes)

        for i in sorted_class_index:
            all_indices = np.where(y_task == i)[0]
            idx = np.random.choice(all_indices, k, replace=False)
            self.x_train.extend(x_task[idx])
            self.y_train.extend(y_task[idx])

            all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
            self.x_valid.extend(x_task[all_indices])
            self.y_valid.extend(y_task[all_indices])
        
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_valid = np.array(self.x_valid)
        self.y_valid = np.array(self.y_valid)

        print("Train: {0}".format(self.x_train.shape[0]))
        print("Valid: {0}".format(self.x_valid.shape[0]))

