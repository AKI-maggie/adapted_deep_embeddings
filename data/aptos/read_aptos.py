import os
import sys
import random
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import json
import cv2

def load_aptos_data(ids):
    imgs = []
    for each in ids:
        print("load {0}".format(each))
        img = imread(each)
        imgs.append(cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC))
    imgs = _process_images(imgs)
    return imgs

def _process_images(images):
    images_np = np.array(images) / 255.0
    return images_np

class Aptos():
    def __init__(self, path):
        self.path = path

        self.img_ids = []
        self.labels = []

        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []

    def read_data(self):
        img_ids = []
        labels = []
        count = 0

        for root, dirs, files in os.walk(self.path):
            for f in files:
                if count >= 4:
                    break
                fpath = os.path.join(root, f)
                if f == "base15.json" \
                    or f == "novel15.json" \
                    or f == "base19.json"\
                    or f == "novel19.json":
                    t1, t2 = self.register_imgs(fpath)
                    img_ids.extend(t1)
                    labels.extend(t2)
                    count += 1
            if count >= 4:
                break
        print("end")
        
        img_ids = np.array(img_ids)
        labels = self.process_labels(labels)

        return img_ids, labels

    def process_labels(self, labels):
        return np.array(labels)

    # domain_option = 
    #   0 - images of all classes 
    #   1 - only images with labels of 0 and 4
    #   2 - only images with labels of 0, 2 and 4
    def register_imgs(self, fpath):
        img_ids = []
        labels = []
        zipped_li = []
        load_imgs_count = [0, 0, 0, 0, 0]

        with open(fpath) as ann:
            meta = json.load(ann)
        
        zipped_li = list(zip(meta['image_names'], meta['image_labels']))
        random.shuffle(zipped_li)

        r_labels = [0, 1, 2, 3, 4]
        for x, y in zipped_li:
            num_y = int(y)
            img_ids.append(x)
            labels.append(num_y)
            load_imgs_count[num_y] += 1
        
        print("Path {0} has loaded: ".format(fpath))
        for i in range(5):
            print("Class {0}: {1} imgs".format(i, load_imgs_count[i]))
        
        return img_ids, labels

    def load_data(self):
        self.img_ids, self.labels = self.read_data()


    def kntl_data_form(self, k1, n1, k2, n2):
        self.load_data()
        print("generate domain")
        self.generate_domain(k1, k2)
        return (self.x_train, self.y_train), (self.x_valid, self.y_valid), \
               (np.array([]), np.array([])), (np.array([]), np.array([]))

    def generate_domain(self, k1, k2):
        all_classes = np.unique(self.labels)

        shuffle = np.random.permutation(self.labels.shape[0])

        x_task, y_task = self.img_ids[shuffle], self.labels[shuffle]

        sorted_class_index = np.sort(all_classes)

        for i in sorted_class_index:
            all_indices = np.where(y_task == i)[0]
            idx = np.random.choice(all_indices, k1, replace=False)
            self.x_train.extend(x_task[idx])
            self.y_train.extend(y_task[idx])

            all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
            idx = np.random.choice(all_indices, 16, replace=False)
            self.x_valid.extend(x_task[idx])
            self.y_valid.extend(y_task[idx])
        
        self.x_train = np.array(self.x_train)
        self.x_valid = np.array(self.x_valid)
        print("Train: {0}".format(self.x_train.shape[0]))
        print("Valid: {0}".format(self.x_valid.shape[0]))
        self.y_train = np.array(self.y_train)
        self.y_valid = np.array(self.y_valid)
        

        self.x_train = load_aptos_data(self.x_train)
        self.x_valid = load_aptos_data(self.x_valid)

