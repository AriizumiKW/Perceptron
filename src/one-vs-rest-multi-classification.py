import numpy as np
import re
import random


def perceptron():
    (weight_vec_1_23, bias_1_23) = perceptron_1_23()
    (weight_vec_2_13, bias_2_13) = perceptron_2_13()
    (weight_vec_3_12, bias_3_12) = perceptron_3_12()
    train_data = readfile('train.data')
    test_data = readfile('test.data')
    successes = 0
    failures = 0
    for (feat_vec, label) in train_data:
        belief_is_class1 = np.matmul(weight_vec_1_23, feat_vec) + bias_1_23
        belief_is_class2 = np.matmul(weight_vec_2_13, feat_vec) + bias_2_13
        belief_is_class3 = np.matmul(weight_vec_3_12, feat_vec) + bias_3_12
        result = 0
        if belief_is_class1 >= belief_is_class2 and belief_is_class1 >= belief_is_class3:
            result = 1
        elif belief_is_class2 >= belief_is_class1 and belief_is_class2 >= belief_is_class3:
            result = 2
        elif belief_is_class3 >= belief_is_class2 and belief_is_class3 >= belief_is_class1:
            result = 3
        if result == label:
            successes += 1
        else:
            failures += 1
    print('overall train accuracy:', successes / (failures + successes))
    successes = 0
    failures = 0
    for (feat_vec, label) in test_data:
        belief_is_class1 = np.matmul(weight_vec_1_23, feat_vec) + bias_1_23
        belief_is_class2 = np.matmul(weight_vec_2_13, feat_vec) + bias_2_13
        belief_is_class3 = np.matmul(weight_vec_3_12, feat_vec) + bias_3_12
        result = 0
        if belief_is_class1>=belief_is_class2 and belief_is_class1>=belief_is_class3:
            result = 1
        elif belief_is_class2>=belief_is_class1 and belief_is_class2>=belief_is_class3:
            result = 2
        elif belief_is_class3>=belief_is_class2 and belief_is_class3>=belief_is_class1:
            result = 3
        if result == label:
            successes += 1
        else:
            failures += 1
    print('overall test accuracy:', successes / (failures + successes))

def perceptron_1_23():
    weight_vec = np.mat(np.array([0.0, 0.0, 0.0, 0.0], dtype='float64'))
    bias = 0.0
    iteration = 1
    train_data = readfile('train.data')
    while iteration <= 20:
        random.shuffle(train_data)  # comment out this line to disable shuffle
        for (feat_vec, label) in train_data:
            a = np.matmul(weight_vec, feat_vec) + bias
            if label == 1:
                y = 1
            else:
                y = -1
            if y*a <= 0: # y=1 for class-1, y=-1 for class-2 and class-3
                weight_vec += feat_vec.T * y
                bias += y
        iteration += 1
    print('train accuracy of 1-vs-23:', test_1_23(weight_vec, bias, 'train.data'))
    print('test accuracy of 1-vs-23:', test_1_23(weight_vec, bias, 'test.data'))
    return (weight_vec, bias)


def perceptron_2_13():
    weight_vec = np.mat(np.array([0.0, 0.0, 0.0, 0.0], dtype='float64'))
    bias = 0.0
    iteration = 1
    train_data = readfile('train.data')
    while iteration <= 20:
        random.shuffle(train_data) # comment out this line to disable shuffle
        for (feat_vec, label) in train_data:
            a = np.matmul(weight_vec, feat_vec) + bias
            if label == 2:
                y = 1
            else:
                y = -1
            if y*a <= 0: # y=1 for class-2, y=-1 for class-1 and class-3
                weight_vec += feat_vec.T * y
                bias += y
        iteration += 1
    print('train accuracy of 2-vs-13:', test_2_13(weight_vec, bias, 'train.data'))
    print('test accuracy of 2-vs-13:', test_2_13(weight_vec, bias, 'test.data'))
    return (weight_vec, bias)


def perceptron_3_12():
    weight_vec = np.mat(np.array([0.0, 0.0, 0.0, 0.0], dtype='float64'))
    bias = 0.0
    iteration = 1
    train_data = readfile('train.data')
    while iteration <= 20:
        random.shuffle(train_data) # comment out this line to disable shuffle
        for (feat_vec, label) in train_data:
            a = np.matmul(weight_vec, feat_vec) + bias
            if label == 3:
                y = 1
            else:
                y = -1
            if y*a <= 0: # y=1 for class-3, y=-1 for class-1 and class-2
                weight_vec += feat_vec.T * y
                bias += y
        iteration += 1
    print('train accuracy of 3-vs-12:', test_3_12(weight_vec, bias, 'train.data'))
    print('test accuracy of 3-vs-12:', test_3_12(weight_vec, bias, 'test.data'))
    return (weight_vec, bias)


def test_1_23(weight_vec, bias, fname):
    test_data = readfile(fname)
    t_positive, t_negative, f_positive, f_negative = 0, 0, 0, 0
    for (feat_vec, label) in test_data:
        a = np.matmul(weight_vec, feat_vec) + bias
        if label == 1: # positive, y=1, class-1
            if a > 0:
                t_positive += 1
            else:
                f_positive += 1
        else: # negative, y=-1, class-2 and class-3
            if a < 0:
                t_negative += 1
            else:
                f_negative += 1
    accuracy = (t_positive + t_negative)/(t_positive + t_negative + f_positive + f_negative)
    return accuracy

def test_2_13(weight_vec, bias, fname):
    test_data = readfile(fname)
    t_positive, t_negative, f_positive, f_negative = 0, 0, 0, 0
    for (feat_vec, label) in test_data:
        a = np.matmul(weight_vec, feat_vec) + bias
        if label == 2: # positive, y=1, class-2
            if a > 0:
                t_positive += 1
            else:
                f_positive += 1
        else: # negative, y=-1, class-1 and class-3
            if a < 0:
                t_negative += 1
            else:
                f_negative += 1
    accuracy = (t_positive + t_negative)/(t_positive + t_negative + f_positive + f_negative)
    return accuracy


def test_3_12(weight_vec, bias, fname):
    test_data = readfile(fname)
    t_positive, t_negative, f_positive, f_negative = 0, 0, 0, 0
    for (feat_vec, label) in test_data:
        a = np.matmul(weight_vec, feat_vec) + bias
        if label == 3: # positive, y=1, class-3
            if a > 0:
                t_positive += 1
            else:
                f_positive += 1
        else: # negative, y=-1, class-1 and class-2
            if a < 0:
                t_negative += 1
            else:
                f_negative += 1
    accuracy = (t_positive + t_negative)/(t_positive + t_negative + f_positive + f_negative)
    return accuracy


def readfile(fname):
    data = []
    with open(fname) as file:
        for line in file:
            the_list = re.split(',', line)
            feat_vec = np.mat(np.array([[float(the_list[0])], [float(the_list[1])], [float(the_list[2])], [float(the_list[3])]]))
            label = the_list[4]
            if label == 'class-1\n':
                data.append((feat_vec, 1))
            elif label == 'class-2\n':
                data.append((feat_vec, 2))
            elif label == 'class-3\n':
                data.append((feat_vec, 3))
    return data


if __name__ == "__main__":
    perceptron()