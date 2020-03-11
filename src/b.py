import numpy as np
import re
import random


def main():
    if_show_you_the_message = False # if you want to know the weight_vector / bias calculated at each iteration
    runs = 500 # how many times should we run to calculate average accuracy
    i = 0
    train_accuracy = 0.0
    test_accuracy = 0.0
    while i < runs:
        tuple = perceptron(if_show_you_the_message)
        train_accuracy += tuple[0]
        test_accuracy += tuple[0]
        i += 1
    print('----------------------------------------------')
    print('train accuracy of (a):' + str(train_accuracy/runs))
    print('test accuracy of (a):' + str(test_accuracy/runs))


def perceptron(message):
    weight_vec = np.mat(np.array([0.0, 0.0, 0.0, 0.0], dtype='float64'))
    bias = 0.0
    iteration = 1
    train_data = readfile('train.data')
    while iteration <= 20:
        random.shuffle(train_data) # comment out this line to disable shuffle
        for (feat_vec, label) in train_data:
            a = np.matmul(weight_vec, feat_vec) + bias
            if label*a <= 0: # label 1 as class-2, label -1 as class-3
                weight_vec += feat_vec.T * label
                bias += label
        iteration += 1
    train_accuracy = test(weight_vec, bias, 'train.data')
    test_accuracy = test(weight_vec, bias, 'test.data')
    if(message):
        print('----------------------------------------------')
        print('weight_vec:', weight_vec)
        print('bias:', bias)
    return (train_accuracy, test_accuracy)


def test(weight_vec, bias, fname):
    test_data = readfile(fname)
    t_positive, t_negative, f_positive, f_negative = 0, 0, 0, 0
    for (feat_vec, label) in test_data:
        a = np.matmul(weight_vec, feat_vec) + bias
        if label == 1: # positive, class-2, labeled as 1
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
    return accuracy


def readfile(fname):
    data = []
    with open(fname) as file:
        for line in file:
            the_list = re.split(',', line)
            feat_vec = np.mat(np.array([[float(the_list[0])], [float(the_list[1])], [float(the_list[2])], [float(the_list[3])]]))
            label = the_list[4]
            if label == 'class-2\n':
                data.append((feat_vec, 1))
            elif label == 'class-3\n':
                data.append((feat_vec, -1))
    return data


if __name__ == "__main__":
    main()