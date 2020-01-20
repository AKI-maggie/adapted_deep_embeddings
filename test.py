import os, sys
from matplotlib.pyplot import imread
import numpy as np

def load(path):
    # variables for training data
    train_annotations = None

    img_path_records = {}

    train_x = []
    train_y = []

    test_x = []

    # gather file information
    print("====================================")
    print("SCANNING FILES AND LOADING TESTING DATA..")
    for root, dirs, files in os.walk(path):
        for f in files:
            f_path = os.path.join(root, f)
            if f == 'train.csv':
                print("Loading annotation file..")
                train_annotations = f_path
            elif f.endswith('.png'):
                # only record paths of training images
                path_splits = f_path.split('/')
                img_dir = path_splits[-2]
                if img_dir == "train_images":
                    print("Loading training image {0}..".format(f))
                    # use id to identify files and record their paths
                    img_id = f.split('.')[0]
                    img_path_records[img_id] = f_path
                else:
                    print("Loading other image {0}..".format(f))
                    # must be test images
                    img = imread(f_path)
                    test_x.append(img)
            else:
                print("Found irrelevant data {0}..".format(f))

    print("====================================")
    print("LOADING TRAINING DATA..")
    # read annotations to load training images and their corresponding labels
    if os.path.exists(path):
        with open(train_annotations) as train_ann:
            # skip the header
            next(train_ann)
            # start loading images
            for line in train_ann:
                contents = line.split()
                # images always have 3 tunnels
                print("Load {0}".format(contents[0]))
                img = imread(img_path_records[contents[0]])
                label = contents[1]
                train_x.append(img)
                train_y.append(label)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)

    return train_x, train_y, test_x

# test run
train_x, train_y, test_x = load('/flush3/zhu041/dataset/aptos/')
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)