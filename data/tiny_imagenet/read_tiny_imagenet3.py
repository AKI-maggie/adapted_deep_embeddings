import os
import sys
import random
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import json
import cv2

# target_path = '/srv/scratch/z5141541/data/aptos/aptos/'

class TinyImageNet():
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2

        self.images1 = []
        self.labels1 = []
        self.images1 = []
        self.labels2 = []

    def load_data(self, target_domain_option, aptos_label_no = 0):
        self.images1, self.labels1 = self.tinyImageNet_load()

        # target_domain_option = 1 --> Both domain use TinyImageNet
        # target_domain_option = 2 --> Read one class of Aptos as a part of the target domain
        # target_domain_option = 3 --> Read all Aptos as target domain resources
        if target_domain_option == 2:
            imgs2, labels2 = self.aptos_load(aptos_label_no)
        elif target_domain_option == 3:
            imgs2, labels2 = self.aptos_load()
        
        self.images2 = imgs2
        self.labels2 = labels2

    # load Aptos data
    # used for testing target source with both Aptos and TinyImageNet
    def aptos_load(self, label_no=-1):
        zipped_li = []
        imgs = []
        labels = []
        aptos_annotation = None
        load_image_count = [0,0,0,0,0]

        # Find the data annotation
        for root, dirs, files in os.walk(self.path2):
            for f in files:
                f_path = os.path.join(root, f)

                # For now, only use base15
                if f == "base15.json":
                    aptos_annotation = f_path
                    break

        if aptos_annotation not None:
            # get image/label pairs
            with open(aptos_annotation) as annotations:
                meta = json.load(annotations)
                zipped_li = list(zip(meta['image_names'], meta['image_labels']))
                random.shuffle(zipped_li)
            
            # read 700 images for each class
            if label_no in range(0, 5):
                # load specified image ids
                for x, y in zipped_li:
                    if int(y) == label_no:
                        if load_image_count[label_no] > 700:
                            break
                        x_image = imread(x)
                        imgs.append(x_image)
                        labels.append(int(y))
                        load_image_count[label_no] += 1
            else:
                # load all image ids
                for x, y in zipped_li:
                    if load_image_count[int(y)] > 700:
                        continue
                    x_image = imread(x)
                    imgs.append(x_image)
                    labels.append(int(y))
                    load_image_count[int(y)] += 1

        else:
            print("Aptos data annotation for base 15 cannot be found!")
            quit()
        
        return imgs, labels

    # load tinyImageNet data
    def tinyImageNet_load(self):
        class_id = 0
        id_to_label = {}
        validation_annotation = None
        validation_images = {}

        images = []
        labels = []

        for root, dirs, files in os.walk(self.path1):
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
                        images.append(img)
                        labels.append(id_to_label[id])
        
        with open(validation_annotations) as val_ann:
            for line in val_ann:
                contents = line.split()
                img = imread(validation_images[contents[0]])
                if len(img.shape) == 2:
                    img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                images.append(img)
                labels.append(id_to_label[contents[1]])

        return images, labels

    # generate dataset for different domain
    # option = 1 --> use single dataset to generate domain
    # option = 2 --> use multiple datasets to generate domain
    def generate_domain(self, k, n, task_no, option, x1, y1, x2=None, y2=None):
        all_classes1 = np.unique(y1)
        all_classes2 = None

        # generate domain with single dataset
        if option == 1:
            task_classes = np.sort(np.random.choice(all_classes1, n, replace=False))
            indices = np.isin(y1, task_classes)
            if task_no == 1:
                self.x_task1, self.y_task1 = x1[indices], y1[indices]
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
                    idx = np.random.choice(all_indices, k, replace=False)
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

            else:   # task_no == 2
                self.x_task2, self.y_task2 = x1[indices], y1[indices]
                shuffle = np.random.permutation(len(self.y_task2))
                self.x_task2, self.y_task2 = self.x_task2[shuffle], self.y_task2[shuffle]

                print('Task 1 Full: {0}'.format(len(self.y_task2)))

                # Force class labels to start from 0 and increment upwards by 1
                sorted_class_indices = np.sort(np.unique(self.y_task2))
                zero_based_classes = np.arange(0, len(sorted_class_indices))
                for i in range(len(self.y_task2)):
                    self.y_task2[i] = zero_based_classes[sorted_class_indices == self.y_task2[i]]

                self.x_train_task2 = []
                self.y_train_task2 = []
                self.x_valid_task2 = []
                self.y_valid_task2 = []

                for i in zero_based_classes:
                    all_indices = np.where(self.y_task2 == i)[0]
                    idx = np.random.choice(all_indices, k, replace=False)
                    self.x_train_task2.extend(self.x_task2[idx])
                    self.y_train_task2.extend(self.y_task2[idx])
                    all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
                    self.x_valid_task2.extend(self.x_task2[all_indices])
                    self.y_valid_task2.extend(self.y_task2[all_indices])

                self.x_train_task2 = np.array(self.x_train_task2)
                self.y_train_task2 = np.array(self.y_train_task2)
                self.x_valid_task2 = np.array(self.x_valid_task2)
                self.y_valid_task2 = np.array(self.y_valid_task2)

                print('Task 2 training: {0}'.format(len(self.x_train_task2)))
                print('Task 2 validation: {0}'.format(len(self.x_valid_task2)))   

        # generate domain with mixture dataset    
        else:       # option == 2
            if y2 not None:
                all_classes2 = np.unique(y2)

                # only use two classes: either it's from Aptos or it's from TinyImagenet


            else:
                print("Should use mixture dataset for task " + str(task_no) + ", but only one dataset found")
                quit()
            


    def kntl_data_form(self, k1, n1, k2, n2, option):
        if option == 1:
            assert k1 < 550 and k2 < 550
        else:
            assert k1 < 550 and k2 < 700    # TinyImage has a maximum of 550 shots for each class, Aptos has 700

        self.load_data(option)

        

        """
        Generate source domain training dataset (TinyImageNet)
        """
        

        """
        Generate target domain training dataset (Aptos)
        """
        if option == 1: # Use TinyImageNet for second training task
            # remove old classes
            tinyImageNet_all_classes = np.delete(tinyImageNet_all_classes, np.where(np.isin(tinyImageNet_all_classes, task1_classes)))
            task2_classes = np.sort(np.random.choice(tinyImageNet_all_classes, n2, replace=False))
            indices = np.isin(self.labels, task2_classes)
            self.x_task2, self.y_task2 = self.images1[indices], self.labels[indices]
            shuffle = np.random.permutation(len(self.y_task2))
            self.x_task2, self.y_task2 = self.x_task2[shuffle], self.y_task2[shuffle]

            
        elif option == 2: # Use one TinyImageNet class and one Aptos class for second training task
            pass
        elif option == 3: # Use Aptos for second training task
            pass
