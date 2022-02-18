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


def load_data(im_path, an_path, load_label):
    annotations = filelist(an_path, '.xml')
    data = []

    for an_path in annotations:
        tree = ET.parse(an_path)
        root = tree.getroot()
        width = int(root.find("./size/width").text)
        height = int(root.find("./size/height").text)
        image_path = root.find("./filename").text

        class_id = 0

        image = cv2.imread(os.path.join(im_path, image_path))
        if load_label:
            # READ ALL 'OBJECT' ELEMENTS FROM .XML FILE
            for obj in tree.findall('object'):
                class_name = obj.find("./name").text
                xmin = int(obj.find("./bndbox/xmin").text)
                xmax = int(obj.find("./bndbox/xmax").text)
                ymin = int(obj.find("./bndbox/ymin").text)
                ymax = int(obj.find("./bndbox/ymax").text)

                if abs(xmax - xmin) > 0.1 * width and abs(ymax - ymin) > 0.1 * height and class_name == "crosswalk":
                    class_id = 1

            data.append({'image': image, 'label': class_id, 'path': image_path})
        else:
            data.append({'image': image, 'path': image_path})

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
    print('accuracy = %.3f' % acc)

    matrix = confusion_matrix(true_labels, pred_labels)
    print(matrix)
    return


def display_dataset_stats(data):
    class_to_num = {}
    for idx, sample in enumerate(data):
        class_id = sample['label']
        if class_id not in class_to_num:
            class_to_num[class_id] = 0
        class_to_num[class_id] += 1

    class_to_num = dict(sorted(class_to_num.items(), key=lambda item: item[0]))
    # print('number of samples for each class:')
    print(class_to_num)


def detect(data):
    for sample in data:
        if sample['label_pred'] == 1:
            gray = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY) #small crosswalk
            _, threshold2 = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY) #big crosswalk
            res = cv2.bitwise_and(threshold, threshold2)

            contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            k = 0

            for i in range(len(contours)):
                #first contour is loaded image
                if i == 0:
                    i = 1

                x, y, w, h = cv2.boundingRect(contours[i])

                approx = cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)
                if (30*40 < cv2.contourArea(contours[i]) < 30000) and (len(approx) == 4 or len(approx) == 5 or len(approx) == 7) \
                        and (0.6 < (w / h) < 1.6) and k == 0:
                    #print detected sign localization
                    print(sample['path'])
                    print(1)
                    print(int(x), int(x + w), int(y), int(y + h))

                    #draw contours
                    cv2.drawContours(sample['image'], [contours[i]], 0, (0, 0, 255), 8)
                    k = k + 1

            if k != 0:
                cv2.imshow(sample['path'], sample['image'])


def main():
    train_im_path = Path('train/images')
    train_an_path = Path('train/annotations')

    data_train = load_data(train_im_path, train_an_path, load_label=True)
    #display_dataset_stats(data_train)

    test_im_path = Path('test/images')
    test_an_path = Path('test/annotations')

    data_test = load_data(test_im_path, test_an_path, load_label=False)
    #display_dataset_stats(data_test)

    #print('learning BoVW')
    learn_bovw(data_train)

    #print('extracting train features')
    data_train = extract_features(data_train)

    #print('training')
    rf = train(data_train)

    #print('extracting test features')
    data_test = extract_features(data_test)

    #print('testing on testing dataset')
    data_test = predict(rf, data_test)
    #evaluate(data_test)

    detect(data_test)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

