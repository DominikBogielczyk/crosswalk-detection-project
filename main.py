import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
        data.append({'image': image, 'label': class_id, 'path': image_path})

    return data

def learn_bovw(data):
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
    for sample in data:
        if sample['desc'] is not None:
            pred = rf.predict(sample['desc'])
            sample['label_pred'] = int(pred)

    return data


def evaluate(data):
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
    #print('accuracy= %.3f' % acc)

    matrix = confusion_matrix(true_labels, pred_labels)
    #print(matrix)
    return

def detect(data):
    for sample in data:
        if sample['label'] == 1:
            img = cv2.imread(os.path.join('test/images/', sample['path']))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # setting threshold of gray image
            _, threshold = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            print(sample['path'])
            k = 0

            i = 0
            for contour in contours:
                #first contour is loaded image
                if i == 0:
                    i = 1

                x, y, w, h = cv2.boundingRect(contour)

                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                if (cv2.contourArea(contour) > 30 * 40) and (len(approx) == 4 or len(approx) == 5 or len(approx) == 7) \
                        and (0.8 < w / h < 1.1):
                    print(int(x - h / 2), int(x + h / 2), int(y - w / 2), int(y + w / 2))
                    cv2.drawContours(img, [contour], 0, (0, 0, 255), 4)
                    k = k + 1

            if k == 0:
                print('Classified but not detected')

            cv2.imshow(sample['path'], img)


def balance_dataset(data, ratio):
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

    #print('learning BoVW')
    #learn_bovw(data_train)

    #print('extracting train features')
    data_train = extract_features(data_train)

    #print('training')
    rf = train(data_train)

    #print('extracting test features')
    data_test = extract_features(data_test)

    #print('testing on testing dataset')
    data_test = predict(rf, data_test)
    evaluate(data_test)

    detect(data_test)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

