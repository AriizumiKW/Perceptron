import numpy as np
import re
import random


def perceptron():
    weight_vec = np.mat(np.array([0.0, 0.0, 0.0, 0.0], dtype='float64'))
    bias = 0.0
    iteration = 1
    train_data = readfile('train.data')
    while iteration <= 20:
        random.shuffle(train_data)
        for (feat_vec, label) in train_data:
            a = np.matmul(weight_vec, feat_vec) + bias
            if label*a <= 0: # label 1 as class-1, label -1 as class-3
                weight_vec += feat_vec.T * label
                bias += label
        iteration += 1
    test(weight_vec, bias)
    print('weight_vec:', weight_vec)
    print('bias:', bias)


def test(weight_vec, bias):
    test_data = readfile('test.data')
    t_positive, t_negative, f_positive, f_negative = 0, 0, 0, 0
    for (feat_vec, label) in test_data:
        a = np.matmul(weight_vec, feat_vec) + bias
        if label == 1: # positive, class-1, labeled as 1
            if a > 0:
                t_positive += 1
            else:
                f_positive += 1
        elif label == -1: # negative, class-3, labeled as -1
            if a < 0:
                t_negative += 1
            else:
                f_negative += 1
    accuracy = (t_positive + t_negative)/(t_positive + t_negative + f_positive + f_negative)
    print('accuracy:', accuracy)


def readfile(fname):
    data = []
    with open(fname) as file:
        for line in file:
            the_list = re.split(',', line)
            feat_vec = np.mat(np.array([[float(the_list[0])], [float(the_list[1])], [float(the_list[2])], [float(the_list[3])]]))
            label = the_list[4]
            if label == 'class-1\n':
                data.append((feat_vec, 1))
            elif label == 'class-3\n':
                data.append((feat_vec, -1))
    return data


if __name__ == "__main__":
    perceptron()