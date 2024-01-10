import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from random import choice
import warnings
from collections import Counter
from kmodes import Kmodes

warnings.filterwarnings("ignore")


def semi_kmodes_method(X, y, k_value):
    # y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # true label
    semi_y = y
    semi_y_new = semi_y[:, np.newaxis]
    with_label_data = np.hstack((X, semi_y_new))
    # print('with_label_data:', with_label_data, type(with_label_data))

    label_set = []
    for i in range(k_value + 1):
        label_set.append([])

    for data in with_label_data:
        for i in range(len(label_set)):
            if data[-1] == i:
                label_set[i].append(data)
                break
            if data[-1] == -1:
                label_set[-1].append(data)
                break
    # print('label_set:', label_set)

    true_label_data = []
    for data in with_label_data:
        if data[-1] != -1:
            true_label_data.append(list(data))

    # print('true_label_data:', true_label_data)

    init_center = []
    for i in range(k_value):
        init_center.append([])

    for i in range(len(label_set) - 1):
        if label_set[i] != []:
            temp = label_set[i][0]
            temp = temp[:-1]
            init_center[i].extend(temp)
        else:
            temp = choice(label_set[-1])
            init_center[i].extend(temp[:-1])

    # init_center = np.array(init_center)
    # print('init_center:', init_center, type(init_center))

    clf = Kmodes(k_value, init_center)
    new_X = clf.continuous_convert_discrete(X)
    # print('len(new_X):',new_X,len(new_X))
    centroids, clusters = clf.find_centers(new_X)
    # print('centroids:',centroids)
    # print('clusters:',clusters)

    y_pred_label = clf.get_labels(clusters, new_X)

    # print('y_pred_label:', y_pred_label,type(y_pred_label),len(y_pred_label))

    # y_pred = KMeans(n_clusters=k_value, init=init_center).fit(X)
    # y_pred_label = y_pred.labels_
    # print('y_pred_label:', y_pred_label)

    class_array = []
    for i in range(k_value):
        class_array.append([])

    # print('class_array:',class_array,type(class_array),len(class_array))
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
        # print('temp:', temp)
        lab = [example[-1] for example in temp]

        count_label = Counter(lab)
        count_label = dict(count_label)
        count_label = sorted(count_label.items(), key=lambda x: x[1], reverse=True)
        # print(count_label, type(count_label))

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
    y = np.array([-1, -1, -1, 1, -1, -1, -1, -1, -1, 0])
    data, label, h = semi_kmodes_method(X, y, 2)
    print(type(label))
    for x, y in zip(data, label):
        print(x, y)
        print('type:x', type(x))
        print('type:y', type(y))


if __name__ == '__main__':
    main()
