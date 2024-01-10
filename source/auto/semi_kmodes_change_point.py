import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from random import choice
import warnings
from collections import Counter
import copy
from kmodes import Kmodes

warnings.filterwarnings("ignore")


def semi_kmodes_method(X, y, num_cluster):
    # y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # true label
    semi_y = copy.deepcopy(y)

    num_class = len(np.unique(semi_y))

    semi_y_new = semi_y[:, np.newaxis]
    with_label_data = np.hstack((X, semi_y_new))
    # print(with_label_data, type(with_label_data))

    label_set = []
    for i in range(num_class):
        label_set.append([])
    #print('label_set:',label_set)
    if len(label_set) == 2:

        for data in with_label_data:

            if data[-1] == 0:
                label_set[-1].append(data)
            else:
                label_set[0].append(data)
    else:
        for data in with_label_data:

           for i in range(len(label_set)):
                if data[-1] == 1:
                    label_set[i].append(data)
                    break
                if data[-1] == 0:
                    label_set[-1].append(data)
                    break
    #print('label_set:', label_set)

    true_label_data = []
    for data in with_label_data:
        if data[-1] != -1:
            true_label_data.append(list(data))

    # print('true_label_data:', true_label_data)

    init_center = []
    for i in range(num_cluster):
        init_center.append([])

    # the difference
    # print("label_set",label_set)
    if len(init_center) < len(label_set) - 1:
        dic = {}
        for i in range(len(label_set)):
            dic[i] = len(label_set[i])
        dic_ = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(init_center)):
            index = dic_[i + 1][0]
            tem_1 = label_set[index][0]
            tem_1 = tem_1[:-1]
            init_center[i].extend(tem_1)

    if len(init_center) == len(label_set) - 1:
        for i in range(len(label_set) - 1):
            tem_2 = label_set[i][0]
            tem_2 = tem_2[:-1]
            init_center[i].extend(tem_2)
    if len(init_center) > len(label_set) - 1:
        length = len(label_set) - 1
        #print(length)
        for i in range(len(init_center)):
            if i < length:
                tem_3 = label_set[i][0]
                tem_3 = tem_3[:-1]
                init_center[i].extend(tem_3)
            else:

                tem_3 = choice(label_set[-1])
                tem_3 = tem_3[:-1]
                init_center[i].extend(tem_3)

    # init_center = np.array(init_center)
    # print('init_center:', init_center, type(init_center))

    clf = Kmodes(num_cluster, init_center)
    new_X = clf.continuous_convert_discrete(X)
    # print('len(new_X):', new_X, len(new_X))
    centroids, clusters = clf.find_centers(new_X)
    # print('centroids:', centroids)
    # print('clusters:', clusters)

    y_pred_label = clf.get_labels(clusters, new_X)

    # y_pred = KMeans(n_clusters=num_cluster, init=init_center).fit(X)
    # y_pred_label = y_pred.labels_
    # print('y_pred_label:', y_pred_label)

    class_array = []
    for i in range(num_cluster):
        class_array.append([])

    for i in range(len(class_array)):
        for j in range(len(y_pred_label)):
            if y_pred_label[j] == i:
                class_array[i].append(with_label_data[j])

    last_label = 0
    changed_dataset = []
    for i in range(len(class_array)):
        temp = []
        for each in class_array[i]:
            temp.append(list(each))
        temp = list(temp)
        # print('temp:all in a cluster', temp)
        lab = [example[-1] for example in temp]

        count_label = Counter(lab)
        count_label = dict(count_label)
        count_label = sorted(count_label.items(), key=lambda x: x[1], reverse=True)
        # print('count_label:',count_label, type(count_label))

        if len(count_label) >= 2:
            if count_label[0][0] == -1:
                last_label = count_label[1][0]
                for data in temp:
                    if data not in true_label_data:
                        data[-1] = last_label
                changed_dataset.extend(temp)
                # print('changed_dataset:', changed_dataset, type(changed_dataset[0]))
            else:
                last_label = count_label[0][0]
                for data in temp:
                    if data not in true_label_data:
                        data[-1] = last_label
                changed_dataset.extend(temp)
        else:
            for data in temp:
                if data not in true_label_data:
                    data[-1] = last_label
            changed_dataset.extend(temp)

    X_list = [example[0:-1] for example in changed_dataset]
    # X_list = np.array(X_list)

    y_list = [example[-1] for example in changed_dataset]
    # y_list = np.array(y_list)

    # print('X_list:', X_list, type(X_list[0]))
    # print('y_list:', y_list, type(y_list))

    return X_list, y_list, init_center


def main():
    X = np.array([[1, 1],
                  [4, 5],
                  [3, 4],
                  [6, 5],
                  [1, 2],
                  [5, 6],
                  [7, 9],
                  [2, 1],
                  [9, 7],
                  [3, 2]])
    print(X.dtype)
    x_train = X[:, :-1]
    print(x_train)
    y = np.array([-1, -1, -1, 1, -1, -1, -1, 0, -1, 0])
    data, label, h = semi_kmodes_method(X, y, 2)
    print(data, label, h)


if __name__ == '__main__':
    main()
