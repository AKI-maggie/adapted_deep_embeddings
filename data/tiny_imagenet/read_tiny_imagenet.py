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
        self.image_ids1 = []
        self.labels1 = []

        self.images2 = []
        self.image_ids2 = []
        self.labels2 = []

        self.x_train_task1 = []
        self.y_train_task1 = []
        self.x_valid_task1 = []
        self.y_valid_task1 = []
        self.x_train_task2 = []
        self.y_train_task2 = []
        self.x_valid_task2 = []
        self.y_valid_task2 = []

    def load_data(self, target_domain_option, aptos_label_no = 0):
        imgs1, img_ids1, labels1 = self.tinyImageNet_load()

        # target_domain_option = 1 --> Both domain use TinyImageNet
        # target_domain_option = 2 --> Read one class of Aptos as a part of the target domain
        # target_domain_option = 3 --> Read all Aptos as target domain resources
        imgs2 = []
        img_ids2 = []
        labels2 = []
        if target_domain_option == 2:
            imgs2, img_ids2, labels2 = self.aptos_load(aptos_label_no)
        elif target_domain_option == 3:
            imgs2, img_ids2, labels2 = self.aptos_load()

        # process into np arrays
        self.images1 = self.process_images(imgs1)
        self.image_ids1 = np.array(img_ids1)
        self.labels1 = self.process_labels(labels1)
        self.images2 = self.process_images(imgs2)
        self.image_ids2 = np.array(img_ids2)
        self.labels2 = self.process_labels(labels2)

        print("Loaded Data Shape:")
        print(self.images1.shape)
        print(self.labels1.shape)
        print(self.images2.shape)
        print(self.labels2.shape)

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        return np.array(labels)

    # load Aptos data
    # used for testing target source with both Aptos and TinyImageNet
    def aptos_load(self, label_no=-1):
        zipped_li = []
        imgs = []
        img_ids = []
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

        if aptos_annotation is not None:
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
                        if load_image_count[label_no] >= 700:
                            break
                        x_image = cv2.resize(imread(x), (64,64), interpolation=cv2.INTER_CUBIC)

                        imgs.append(x_image)
                        img_ids.append(x)
                        labels.append(int(y))
                        load_image_count[label_no] += 1
            else:
                # load all image ids
                for x, y in zipped_li:
                    if load_image_count[int(y)] >= 300:
                        continue
                    x_image = cv2.resize(imread(x), (64,64), interpolation=cv2.INTER_CUBIC)
                    imgs.append(x_image)
                    img_ids.append(x)
                    labels.append(int(y))
                    load_image_count[int(y)] += 1

        else:
            print("Aptos data annotation for base 15 cannot be found!")
            quit()
        
        print("Count:")
        for each in load_image_count:
            print(each)
        return imgs, img_ids, labels

    # load tinyImageNet data
    def tinyImageNet_load(self):
        class_id = 0
        id_to_label = {}
        validation_annotation = None
        validation_images = {}

        images = []
        img_ids = []
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
                        img_ids.append(path)
                        labels.append(id_to_label[id])
        
        with open(validation_annotations) as val_ann:
            for line in val_ann:
                contents = line.split()
                img = imread(validation_images[contents[0]])
                if len(img.shape) == 2:
                    img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                images.append(img)
                img_ids.append(validation_images[contents[0]])
                labels.append(id_to_label[contents[1]])

        return images, img_ids, labels

    # generate dataset for different domain
    # option = 1 --> Both domain use TinyImageNet
    # option = 2 --> Read one class of Aptos as a part of the target domain
    # option = 3 --> Read all Aptos as target domain resources
    def generate_domain(self, k, n, task_no, option, x1, y1, z1, x2=None, y2=None, z2=None):
        all_classes1 = np.unique(y1)
        all_classes2 = None

        # print(y2)

        # generate domain with single dataset
        if option == 1:
            task_classes = np.sort(np.random.choice(all_classes1, n, replace=False))
            indices = np.isin(y1, task_classes)
            if task_no == 1:
                self.x_task1, self.y_task1, z = x1[indices], y1[indices], z1[indices]
                shuffle = np.random.permutation(len(self.y_task1))
                self.x_task1, self.y_task1, z = self.x_task1[shuffle], self.y_task1[shuffle], z[shuffle]

                print('Task 1 Full: {0}'.format(len(self.y_task1)))

                # Force class labels to start from 0 and increment upwards by 1
                sorted_class_indices = np.sort(np.unique(self.y_task1))
                zero_based_classes = np.arange(0, len(sorted_class_indices))
                for i in range(len(self.y_task1)):
                    self.y_task1[i] = zero_based_classes[sorted_class_indices == self.y_task1[i]]

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
                self.x_task2, self.y_task2, z = x1[indices], y1[indices], z1[indices]
                shuffle = np.random.permutation(len(self.y_task2))
                self.x_task2, self.y_task2, z = self.x_task2[shuffle], self.y_task2[shuffle], z[shuffle]

                print(z[:5])
                print(self.y_task2[:5])

                print('Task 2 Full: {0}'.format(len(self.y_task2)))

                zero_based_classes = np.sort(np.unique(self.y_task2))

                for i in zero_based_classes:
                    print("Finding for "+str(i))
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

        # generate domain with mixture dataset (only used for task 2 training)
        else:       # option == 2
            if y2 is not None:
                indices = np.isin(y1, [all_classes1[0]])
                # extend tiny images
                tiny_img_x, tiny_img_y, z = x1[indices], y1[indices], z1[indices]
                shuffle = np.random.permutation(len(tiny_img_y))
                tiny_img_x, tiny_img_y, z = tiny_img_x[shuffle], np.full((tiny_img_y.shape[0]), 0), z[shuffle]

                all_indices = np.where(tiny_img_y == 0)[0]
                for each in all_indices:
                    print(each)
                idx = np.random.choice(all_indices, k, replace=False)
                print(len(idx))
                self.x_train_task2.extend(tiny_img_x[idx])
                self.y_train_task2.extend(tiny_img_y[idx])

                all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
                idx = np.random.choice(all_indices, k, replace=False)
                self.x_valid_task2.extend(tiny_img_x[idx])
                self.y_valid_task2.extend(tiny_img_y[idx])

                # extend aptos images
                aptos_img_x, aptos_img_y, z = x2, y2, z2
                shuffle = np.random.permutation(len(aptos_img_y))
                aptos_img_x, aptos_img_y, z = aptos_img_x[shuffle], np.full((aptos_img_y.shape[0]), 1), z[shuffle]

                print(aptos_img_y.shape)
                all_indices = range(0, aptos_img_y.shape[0])
                idx = np.random.choice(all_indices, k, replace=False)
                self.x_train_task2.extend(aptos_img_x[idx])
                self.y_train_task2.extend(aptos_img_y[idx])

                all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
                idx = np.random.choice(all_indices, k, replace=False)
                self.x_valid_task2.extend(aptos_img_x[idx])
                self.y_valid_task2.extend(aptos_img_y[idx])

                self.x_train_task2 = np.array(self.x_train_task2)
                self.y_train_task2 = np.array(self.y_train_task2)
                self.x_valid_task2 = np.array(self.x_valid_task2)
                self.y_valid_task2 = np.array(self.y_valid_task2)

                print(self.x_train_task1.shape)
                print(self.y_train_task1.shape)
                print(self.x_valid_task1.shape)
                print(self.y_valid_task1.shape)
                print(self.x_train_task2.shape)
                print(self.y_train_task2.shape)
                print(self.x_valid_task2.shape)
                print(self.y_valid_task2.shape)

                # shuffle
                shuffle = np.random.permutation(len(self.x_train_task2))
                shuffle2 = np.random.permutation(len(self.x_valid_task2))

                self.x_train_task2, self.y_train_task2 = self.x_train_task2[shuffle], self.y_train_task2[shuffle]
                self.x_valid_task2, self.y_valid_task2 = self.x_valid_task2[shuffle2], self.y_valid_task2[shuffle]

                print('Task 2 training: {0}'.format(len(self.x_train_task2)))
                print('Task 2 validation: {0}'.format(len(self.x_valid_task2)))             

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
        self.generate_domain(k1, n1, 1, 1, self.images1, self.labels1, self.image_ids1)

        """
        Generate target domain training dataset (Aptos)
        """
        if option == 1:
            self.generate_domain(k2, n2, 2, 1, self.images1, self.labels1, self.image_ids1)
        elif option == 2:
            self.generate_domain(k2, n2, 2, 2, self.images1, self.labels1, self.image_ids1, self.images2, self.labels2, self.image_ids2)
        else:
            self.generate_domain(k2, n2, 2, 1, self.images2, self.labels2, self.image_ids2)

        print(self.x_train_task2.shape)
        print(self.y_train_task2.shape)
        print(self.x_valid_task2.shape)
        print(self.y_valid_task2.shape)

        print('k = {0}, n = {1}'.format(k2, n1))
        print('Task 2 training: {0}'.format(len(self.x_train_task2)))
        print('Task 2 test: {0}\n'.format(len(self.x_valid_task2)))

        return (self.x_train_task1, self.y_train_task1), (self.x_valid_task1, self.y_valid_task1), (self.x_train_task2, self.y_train_task2), (self.x_valid_task2, self.y_valid_task2)

