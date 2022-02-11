import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
                                                files in os.walk(root) for f in files if f.endswith(file_type)]


def load_data(im_path, an_path):
    annotations = filelist(an_path, '.xml')
    data = []

    for an_path in annotations:
        tree = ET.parse(an_path)
        root = tree.getroot()
        width = int(root.find("./size/width").text)
        height = int(root.find("./size/height").text)
        image_path = root.find("./filename").text

        class_id = 0

        # READ ALL 'OBJECT' ELEMENTS FROM .XML FILE
        for obj in tree.findall('object'):
            class_name = obj.find("./name").text
            xmin = int(obj.find("./bndbox/xmin").text)
            xmax = int(obj.find("./bndbox/xmax").text)
            ymin = int(obj.find("./bndbox/ymin").text)
            ymax = int(obj.find("./bndbox/ymax").text)

            if abs(xmax - xmin) > 0.1 * width and abs(ymax - ymin) > 0.1 * height and class_name == "crosswalk":
                class_id = 1

        image = cv2.imread(os.path.join(im_path, image_path))
        data.append({'image': image, 'label': class_id})

    return data

def learn_bovw(data):
    #Learns BoVW dictionary and saves it as "voc.npy" file.

    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)


def extract_features(data):
    #Extracts features for given data and saves it as "desc" entry.

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)

    for sample in data:
        # compute descriptor and add it as "desc" entry in sample
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        sample['desc'] = desc

    return data

def train(data):
    #Trains Random Forest classifier.

    # train random forest model and return it from function.
    descs = []
    labels = []

    for sample in data:
        if sample['desc'] is not None:
            descs.append(sample['desc'].squeeze(0))
            labels.append(sample['label'])

    clf = RandomForestClassifier()
    clf.fit(descs, labels)

    return clf

def predict(rf, data):
    #Predicts labels given a model and saves them as "label_pred" (int) entry for each sample.

    # perform prediction using trained model and add results as "label_pred" (int) entry in sample

    for sample in data:
        if sample['desc'] is not None:
            pred = rf.predict(sample['desc'])
            sample['label_pred'] = int(pred)

    return data

def evaluate(data):
    #Evaluates results of classification.

    # evaluate classification results and print statistics
    pred_labels = []
    true_labels = []

    l = 0
    m = 0
    for sample in data:
        if sample['desc'] is not None:
            pred_labels.append(sample['label_pred'])
            true_labels.append(sample['label'])
            if sample['label'] == sample['label_pred']:
                l = l + 1
            else:
                m = m + 1
    acc = l/(l+m)
    print('accuracy= %.3f' % acc)

    matrix = confusion_matrix(true_labels, pred_labels)
    print(matrix)

    # this function does not return anything
    return

def balance_dataset(data, ratio):
    #Subsamples dataset according to ratio.

    sampled_data = random.sample(data, int(ratio * len(data)))

    return sampled_data

def main():
    train_im_path = Path('train/images')
    train_an_path = Path('train/annotations')

    data_train = load_data(train_im_path, train_an_path)
    data_train = balance_dataset(data_train, 1.0)

    test_im_path = Path('test/images')
    test_an_path = Path('test/annotations')

    data_test = load_data(test_im_path, test_an_path)
    data_test = balance_dataset(data_test, 1.0)

    print('learning BoVW')
    #learn_bovw(data_train)

    print('extracting train features')
    data_train = extract_features(data_train)

    print('training')
    rf = train(data_train)

    print('extracting test features')
    data_test = extract_features(data_test)

    print('testing on testing dataset')
    data_test = predict(rf, data_test)
    evaluate(data_test)

if __name__ == '__main__':
    main()

