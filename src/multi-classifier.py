import numpy as np
import re
import random


def main():
    # ----------------------------------------------------------------------
    lam = 0  # set the lambda here for L2-regularisation (set to zero to disable L2-regularisation)
    if_shuffle = True  # if shuffle or not
    max_iteration_num = 20  # number of iterations
    runs = 500  # how many times should we run to calculate average accuracy
    # ----------------------------------------------------------------------
    i = 0
    accuracy_1vs23_train = 0.0
    accuracy_1vs23_test = 0.0
    accuracy_2vs13_train = 0.0
    accuracy_2vs13_test = 0.0
    accuracy_3vs12_train = 0.0
    accuracy_3vs12_test = 0.0
    accuracy_overall_train = 0.0
    accuracy_overall_test = 0.0
    while i < runs:
        the_tuple = perceptron(lam, if_shuffle, max_iteration_num)
        accuracy_1vs23_train += the_tuple[0]
        accuracy_1vs23_test += the_tuple[1]
        accuracy_2vs13_train += the_tuple[2]
        accuracy_2vs13_test += the_tuple[3]
        accuracy_3vs12_train += the_tuple[4]
        accuracy_3vs12_test += the_tuple[5]
        accuracy_overall_train += the_tuple[6]
        accuracy_overall_test += the_tuple[7]
        i += 1
    print("train accuracy 1-vs-23:" + str(accuracy_1vs23_train / runs))
    print("test accuracy 1-vs-23:" + str(accuracy_1vs23_test / runs))
    print("train accuracy 2-vs-13:" + str(accuracy_2vs13_train / runs))
    print("test accuracy 2-vs-13:" + str(accuracy_2vs13_test / runs))
    print("train accuracy 3-vs-12:" + str(accuracy_3vs12_train / runs))
    print("test accuracy 3-vs-12:" + str(accuracy_3vs12_test / runs))
    print("overall train accuracy:" + str(accuracy_overall_train / runs))
    print("overall test accuracy:" + str(accuracy_overall_test / runs))


def perceptron(lam, shuffle, iter_num):
    (weight_vec_1_23, bias_1_23) = perceptron_1_23(lam, shuffle, iter_num)
    (weight_vec_2_13, bias_2_13) = perceptron_2_13(lam, shuffle, iter_num)
    (weight_vec_3_12, bias_3_12) = perceptron_3_12(lam, shuffle, iter_num)
    accuracy_1vs23_train = test_1_23(weight_vec_1_23, bias_1_23, 'train.data')
    accuracy_1vs23_test = test_1_23(weight_vec_1_23, bias_1_23, 'test.data')
    accuracy_2vs13_train = test_2_13(weight_vec_2_13, bias_2_13, 'train.data')
    accuracy_2vs13_test = test_2_13(weight_vec_2_13, bias_2_13, 'test.data')
    accuracy_3vs12_train = test_3_12(weight_vec_3_12, bias_3_12, 'train.data')
    accuracy_3vs12_test = test_3_12(weight_vec_3_12, bias_3_12, 'test.data')
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
    accuracy_overall_train = successes / (failures + successes)
    successes = 0
    failures = 0
    for (feat_vec, label) in test_data:
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
    accuracy_overall_test = successes / (failures + successes)
    return (accuracy_1vs23_train, accuracy_1vs23_test, accuracy_2vs13_train, accuracy_2vs13_test, accuracy_3vs12_train,
            accuracy_3vs12_test, accuracy_overall_train, accuracy_overall_test)

def perceptron_1_23(lam, shuffle, iter_num):
    weight_vec = np.mat(np.array([0.0, 0.0, 0.0, 0.0], np.longdouble))
    bias = 0.0
    iteration = 1
    train_data = readfile('train.data')
    while iteration <= iter_num:
        if shuffle:
            random.shuffle(train_data)
        for (feat_vec, label) in train_data:
            a = np.matmul(weight_vec, feat_vec) + bias
            if label == 1:
                y = 1
            else:
                y = -1
            if y*a <= 0:  # y=1 for class-1, y=-1 for class-2 and class-3
                weight_vec = weight_vec + (feat_vec.T * y) - (2 * lam * weight_vec)
                bias = bias + y
                scalingDown(weight_vec, bias)
        iteration += 1
    return (weight_vec, bias)


def perceptron_2_13(lam, shuffle, iter_num):
    weight_vec = np.mat(np.array([0.0, 0.0, 0.0, 0.0], np.longdouble))
    bias = 0.0
    iteration = 1
    train_data = readfile('train.data')
    while iteration <= iter_num:
        if shuffle:
            random.shuffle(train_data)
        for (feat_vec, label) in train_data:
            a = np.matmul(weight_vec, feat_vec) + bias
            if label == 2:
                y = 1
            else:
                y = -1
            if y*a <= 0:  # y=1 for class-2, y=-1 for class-1 and class-3
                weight_vec = weight_vec + (feat_vec.T * y) - (2 * lam * weight_vec)
                bias = bias + y
                scalingDown(weight_vec, bias)
        iteration += 1
    return (weight_vec, bias)


def perceptron_3_12(lam, shuffle, iter_num):
    weight_vec = np.mat(np.array([0.0, 0.0, 0.0, 0.0], np.longdouble))
    bias = 0.0
    iteration = 1
    train_data = readfile('train.data')
    while iteration <= iter_num:
        if shuffle:
            random.shuffle(train_data)
        for (feat_vec, label) in train_data:
            a = np.matmul(weight_vec, feat_vec) + bias
            if label == 3:
                y = 1
            else:
                y = -1
            if y*a <= 0:  # y=1 for class-3, y=-1 for class-1 and class-2
                weight_vec = weight_vec + (feat_vec.T * y) - (2 * lam * weight_vec)
                bias = bias + y
                scalingDown(weight_vec, bias)
        iteration += 1
    return (weight_vec, bias)


def test_1_23(weight_vec, bias, fname):
    test_data = readfile(fname)
    t_positive, t_negative, f_positive, f_negative = 0, 0, 0, 0
    for (feat_vec, label) in test_data:
        a = np.matmul(weight_vec, feat_vec) + bias
        if label == 1:  # positive, y=1, class-1
            if a > 0:
                t_positive += 1
            else:
                f_positive += 1
        else:  # negative, y=-1, class-2 and class-3
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
        if label == 2:  # positive, y=1, class-2
            if a > 0:
                t_positive += 1
            else:
                f_positive += 1
        else:  # negative, y=-1, class-1 and class-3
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
        if label == 3:  # positive, y=1, class-3
            if a > 0:
                t_positive += 1
            else:
                f_positive += 1
        else:  # negative, y=-1, class-1 and class-2
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
            feat_vec = np.mat(np.array([[float(the_list[0])], [float(the_list[1])], [float(the_list[2])], [float(the_list[3])]]), dtype='float64')
            label = the_list[4]
            if label == 'class-1\n':
                data.append((feat_vec, 1))
            elif label == 'class-2\n':
                data.append((feat_vec, 2))
            elif label == 'class-3\n':
                data.append((feat_vec, 3))
    return data


def scalingDown(weight_vec, bias):
    if weight_vec[0,0] >= 1e+100:
        weight_vec /= 1e+100
        bias /= 1e+100


if __name__ == "__main__":
    main()