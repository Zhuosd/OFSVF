import numpy as np
import pandas as pd
import time
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import copy
from collections import Counter
from sklearn.cluster import KMeans
import random
from semi_kmodes_change_point import semi_kmodes_method
from rho_multi_delta import Rho_multi_Delta
from change_ponit import Change_point
from kmodes import Kmodes
from original_drift_detection import Drift_detection

# VFDT node class
class VfdtNode:
    def __init__(self, possible_split_features, k_value):  # __init__构造方法，初始化
        """
        nijk: statistics of feature i, value j, class k
        :list possible_split_features: features
        """
        self.parent = None
        self.brother = None
        self.left_child = None  # left_child
        self.right_child = None  # right_child
        self.split_feature = None
        self.split_value = None
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {f: {} for f in possible_split_features}
        # nijk={'a':{},'b':{},'c':{}}
        self.possible_split_features = possible_split_features
        self.k_value = k_value
        self.data_array = []
        self.part_label = []
        self.cluster_info_hist = None
        self.cluster_info_new = None
        self.center_hist = None
        self.center_new = None
        self.intervals_hist = None
        self.intervals_new = None

    def add_children(self, split_feature, split_value, left, right):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self
        left.brother = right
        right.brother = left

        self.nijk.clear()  # reset stats
        if isinstance(split_value, list):
            left_value = split_value[0]
            right_value = split_value[1]
            # discrete split value list's length = 1, stop splitting

            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
                new_features = [None if f == split_feature else f for f in right.possible_split_features]
                right.possible_split_features = new_features

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def sort_example_train(self, x, y):
        #print("-----++++", y)
        if self.is_leaf():
            self.data_array.append(x)
            self.part_label.append(y)
            self.new_examples_seen += 1
            return self
        else:

            index = self.possible_split_features.index(self.split_feature)
            value = x[index]
            #print('1',value)
            split_value = self.split_value

            if isinstance(split_value, list):  # discrete value
                if value in split_value[0]:  # isinstance()
                    return self.left_child.sort_example_train(x, y)
                else:
                    return self.right_child.sort_example_train(x, y)
            else:  # continuous value
                if value <= split_value:
                    return self.left_child.sort_example_train(x, y)
                else:
                    return self.right_child.sort_example_train(x, y)

    def sort_example(self, x):
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)

            value = x[index]

            split_value = self.split_value
            if isinstance(split_value, list):
                # discrete value
                if value in split_value[0]:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)
            else:  # continuous value
                if value <= split_value:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)

    # the most frequent class
    def most_frequent(self):
        try:

            prediction = max(self.class_frequency, key=self.class_frequency.get)

        except ValueError:
            # if self.class_frequency dict is empty, go back to parent
            #print('parent',self.parent)
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return prediction

    # update leaf stats in order to calculate G()
    def update_stats(self, x, y):
        feats = self.possible_split_features  # possible_split_features
        #print('feats:',feats)
        nijk = self.nijk
        if not bool(nijk):
            nijk = {f: {} for f in feats}

        iterator = [f for f in feats if f is not None]
        # print('iterator',iterator)
        for i in iterator:
            value = x[feats.index(i)]
            # print("iter",iterator)
            # print("x是",x)
            # print('value:',value)
            if value not in nijk[i]:
                nijk[i][value] = {y: 1}
            else:
                try:
                    nijk[i][value][y] += 1
                except KeyError:  # keyError
                    nijk[i][value][y] = 1

        class_frequency = self.class_frequency
        self.total_examples_seen += 1
        # self.new_examples_seen += 1
        #print('class_frequency',class_frequency)
        # 记录当前叶子节点的类别分布
        try:
            class_frequency[y] += 1
        except KeyError:  # keyError
            class_frequency[y] = 1

    def check_not_splitting(self):
        # compute gini index for not splitting
        X0 = 1
        class_frequency = self.class_frequency
        # print(class_frequency) #{'no':179 , 'yes':21}
        # print(len(class_frequency)) # len=2
        # print(class_frequency.values()) #dict_values([179,21])
        n = sum(class_frequency.values())
        # print(class_frequency.items()) dict_items([('no', 179), ('yes', 21)])
        for j, k in class_frequency.items():
            X0 -= (k / n) ** 2
        return X0

    def attempt_split(self, delta, nmin, tau):  # use Hoeffding tree model to test node split, return the split feature
        if self.new_examples_seen < nmin:  #
            return None  # return None

        class_frequency = self.class_frequency
        if len(class_frequency) == 1:  #
            return None

        self.new_examples_seen = 0
        self.data_array = []
        self.part_label = []
        nijk = self.nijk
        min = 1
        second_min = 1
        Xa = ''
        split_value = None
        #print(self.possible_split_features) #possible_s_f=['age','job',...'poutcome']
        for feature in self.possible_split_features:
            if feature is not None:
                #print('feature',feature)
                #print("nijk：",nijk)#{'age':{44:{'no':6},...35{'yes':8}...},...'job':{'unem':{'no':9}...}}}
                njk = nijk[feature]
                #print("njk：", njk )
                # {'married': {'no': 117, 'yes': 13}, 'single': {'no': 42, 'yes': 6}, 'divorced': {'no': 18, 'yes': 4}}
                gini, value = self.gini(njk, class_frequency)
                if gini < min:
                    min = gini
                    Xa = feature
                    split_value = value
                    # print(split_value)
                elif min < gini < second_min:
                    second_min = gini

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting()
        if min < g_X0:
            if second_min - min > epsilon:
                # print('1 node split')
                return [Xa, split_value]
            elif second_min - min < epsilon < tau:  # ΔH<e<t
                # print('2 node split')
                return [Xa, split_value]
            else:
                return None
        return None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log2(len(self.class_frequency))
        return np.sqrt(R * R * np.log(1 / delta) / (2 * n))

    def gini(self, njk, class_frequency):
        # gini(D) = 1 - Sum(pi^2)
        # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)

        D = self.total_examples_seen
        m1 = 1  # minimum gini
        # m2 = 1  # second minimum gini
        Xa_value = None
        # print(njk)#{'married': {'no': 111, 'yes': 8}, 'single': {'yes': 8, 'no': 45}, 'divorced': {'no': 26, 'yes': 2}}
        feature_values = list(njk.keys())  # list() is essential
        #print(feature_values) #['married', 'single', 'divorced']
        if not isinstance(feature_values[0], str):  # numeric  feature values
            sort = np.array(sorted(feature_values))  #
            # print(sort)#sorted
            split = (sort[0:-1] + sort[1:]) / 2  # vectorized computation, like in R
            # print(split)
            # print(sort[0:-1],sort[1:])
            D1_class_frequency = {j: 0 for j in class_frequency.keys()}  #
            # print(D1_class_frequency)#{'no': 0, 'yes': 0}
            for index in range(len(split)):
                nk = njk[sort[index]]
                # print(nk)
                # nk={'no':71,'yes':3}
                for j in nk:
                    D1_class_frequency[j] += nk[j]
                    # D1_class_frequency=D1_class_frequency + nk[j]
                    # print(D1_class_frequency)#{'no':14,'yes':2}
                    # print(D1_class_frequency.values()) #dict_values([14,2])
                D1 = sum(D1_class_frequency.values())  # sum=16
                D2 = D - D1  # D = self.total_examples_seen=nmin = 200
                g_d1 = 1
                g_d2 = 1

                D2_class_frequency = {}
                for key, value in class_frequency.items():
                    #print(class_frequency)
                    #print(D1_class_frequency)
                    if key in D1_class_frequency:
                        D2_class_frequency[key] = value - D1_class_frequency[key]
                        # print(D2_class_frequency)
                        # print('\n')
                    else:
                        D2_class_frequency[key] = value

                for key, v in D1_class_frequency.items():
                    g_d1 -= (v / D1) ** 2
                for key, v in D2_class_frequency.items():
                    g_d2 -= (v / D2) ** 2
                g = g_d1 * D1 / D + g_d2 * D2 / D
                if g < m1:  # m1=1
                    m1 = g
                    Xa_value = split[index]
                    # elif m1 < g < m2:
                # m2 = g
            return [m1, Xa_value]

        else:  # discrete feature_values
            length = len(njk)
            # print(njk)#{'married': {'no': 111, 'yes': 8}, 'single': {'yes': 8, 'no': 45}, 'divorced': {'no': 26, 'yes': 2}}
            if length > 10:  # too many discrete feature values, estimate
                for j, k in njk.items():

                    D1 = sum(k.values())  # {'no': 111, 'yes': 8}
                    D2 = D - D1  # D=200
                    g_d1 = 1
                    g_d2 = 1

                    D2_class_frequency = {}
                    for key, value in class_frequency.items():  # class_frequency={'no':183,'yes':17}
                        # print(class_frequency)
                        if key in k:  # n=200
                            # print(key)
                            # print(k)
                            D2_class_frequency[key] = value - k[key]
                            # {'no':183-40,'yes':17-3}
                        else:
                            D2_class_frequency[key] = value
                    for key, v in k.items():
                        g_d1 -= (v / D1) ** 2

                    if D2 != 0:
                        for key, v in D2_class_frequency.items():
                            g_d2 -= (v / D2) ** 2
                    g = g_d1 * D1 / D + g_d2 * D2 / D
                    if g < m1:
                        m1 = g
                        Xa_value = j
                    # elif m1 < g < m2:
                    # m2 = g
                # print(feature_values)
                # print(Xa_value)
                right = list(np.setdiff1d(feature_values, Xa_value))
                # print(right)

            else:  # fewer discrete feature values, get combinations
                comb = self.select_combinations(feature_values)
                # print(type(comb)) <class 'list'>
                for i in comb:
                    left = list(i)
                    # print(left)
                    D1_class_frequency = {key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {key: 0 for key in class_frequency.keys()}
                    # print(D1_class_frequency,D2_class_frequency) #{'no': 0, 'yes': 0} {'no': 0, 'yes': 0}
                    for j, k in njk.items():
                        for key, value in class_frequency.items():
                            # print(class_frequency)#{'no': 189, 'yes': 11}
                            if j in left:
                                if key in k:
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    D1 = sum(D1_class_frequency.values())
                    D2 = D - D1
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v / D1) ** 2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v / D2) ** 2
                    g = g_d1 * D1 / D + g_d2 * D2 / D
                    if g < m1:
                        m1 = g
                        Xa_value = left
                    # elif m1 < g < m2:
                    # m2 = g

                # print(len(comb))
                right = list(np.setdiff1d(feature_values, Xa_value))
            return [m1, [Xa_value, right]]

            # divide values into two groups, return the combination of left groups

    def select_combinations(self, feature_values):
        combination = []
        e = len(feature_values)
        if e % 2 == 0:
            end = int(e / 2)
            for i in range(1, end + 1):
                if i == end:
                    cmb = list(combinations(feature_values, i))
                    enough = int(len(cmb) / 2)
                    combination.extend(cmb[:enough])
                else:
                    combination.extend(combinations(feature_values, i))
        else:
            end = int((e - 1) / 2)
            for i in range(1, end + 1):
                combination.extend(combinations(feature_values, i))
        # print(combination)#[('married',), ('divorced',), ('single',)]
        return combination


# very fast decision tree class, i.e. hoeffding tree #i.e.= in other words
class Vfdt:
    def __init__(self, features, delta=0.0000001, nmin=20, tau=0.05, k_value=2):
        """
        :features: list of dataset features
        :delta: used to compute hoeffding bound, error rate
        :nmin: to limit the G computations
        :tau: to deal with ties
        """
        self.features = features
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        self.k_value = k_value
        self.root = VfdtNode(features, k_value)
        # self.root = VfdtNode(features)
        self.n_examples_processed = 0

    # update the tree by adding training example
    def update(self, x, y):
        self.n_examples_processed += 1
        node = self.root.sort_example_train(x, y)
        # print('..',node.new_examples_seen)
        # print(self.nmin)
        if node.new_examples_seen == self.nmin:
            pre_x_train = np.array(node.data_array)
            #print("pre_x_train:", pre_x_train)
            pre_y_train = np.array(node.part_label)
            #print("pre_y_train:", pre_y_train)
            #print('node.k_value',node.k_value)
            x_train, y_train, unuse_center = semi_kmodes_method(pre_x_train, pre_y_train, node.k_value)
            #print('center',unuse_center)
            #print("y_train:", y_train)
            for x_1, y_1 in zip(x_train, y_train):
                #("x_1",x_1, "y_1",y_1)
                node.update_stats(x_1, y_1)

            result = node.attempt_split(self.delta, self.nmin, self.tau)
            #print('result', result)
            number_cluster = self.deter_num_cluster(pre_x_train)
            #print(number_cluster)

            unuse_x_train, unuse_y_train, use_center = semi_kmodes_method(pre_x_train, pre_y_train, number_cluster)
            #print('unuse_center',unuse_center)
            if result is None:
                if node.cluster_info_hist is None:
                    node.center_hist, node.cluster_info_hist, node.intervals_hist = \
                        self.record_concept_clusters(node, number_cluster, use_center, pre_x_train)

                # else:
                #     brother_node = node.brother
                #     self.detect_concept_drift(node, brother_node, pre_x_train, number_cluster, use_center)

            if result is not None:
                feature = result[0]
                value = result[1]
                self.node_split(node, feature, value)

    def record_concept_clusters(self, node, number_cluster, use_center, pre_x_train):
        clf = Kmodes(number_cluster, use_center)
        X = clf.continuous_convert_discrete(pre_x_train)
        node.center_hist, node.cluster_info_hist = clf.find_centers(X)
        node.intervals_hist = clf.find_intervals_info(pre_x_train, node.center_hist)

        return node.center_hist, node.cluster_info_hist, node.intervals_hist



    def deter_num_cluster(self, pre_x_train):

        rho_delta = Rho_multi_Delta(pre_x_train)

        rho_multi_delta = rho_delta.cluster()

        change_point = Change_point()
        number_cluster = change_point.num_cluster(rho_multi_delta)

        return number_cluster

    def node_split(self, node, split_feature, split_value):
        features = node.possible_split_features
        k_value = self.k_value

        left = VfdtNode(features, k_value)
        right = VfdtNode(features, k_value)
        # print(left)
        # print(right)
        # left = VfdtNode(features)
        # right = VfdtNode(features)
        node.add_children(split_feature, split_value, left, right)

    # predict test example's classification
    def predict(self, x_test):
        #print("x_test",x_test)

        prediction = []  # ndarry
        if isinstance(x_test, np.ndarray) or isinstance(x_test, list):

            for x in x_test:
                #print('x',x)
                leaf = self.root.sort_example(x)
                #print(self.root)

                #print('leaf:',leaf.most_frequent())
                #print(type(leaf))
                prediction.append(leaf.most_frequent())

            return prediction
        else:
            leaf = self.root.sort_example(x_test)
            return leaf.most_frequent()

    def print_tree(self, node):
        if node.is_leaf():
            print('Leaf')
        else:
            #print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)


def calc_metrics(y_test, y_pred, row_name):
    accuracy = accuracy_score(y_test, y_pred)
    metrics = list(precision_recall_fscore_support(y_test, y_pred, average='weighted',
                                                   labels=np.unique(y_pred)))

    metrics = pd.DataFrame({'accuracy': accuracy, 'precision': metrics[0], 'recall': metrics[1],
                            'f1': metrics[2]}, index=[row_name])

    return metrics


def SSC_DensityPeaks_SVC_clu(train, label_train, t, nneigh): # , test, label_test #, clf, n, decay_choice, contribute_error_rate

    data = []
    label_data = []
    label_U = []

    train_x = pd.DataFrame(train)
    title = list(train_x.columns.values)
    features = title[:]
    tree = Vfdt(features, delta=0.001, nmin=20, tau=0.5)

    #print(t)
    data.append(train[t-1,]) # t-1
    data = np.array(data)
    data = data.reshape(data.shape[1],data.shape[2])

    label_data.append(label_train[t-1,]) # t-1
    label_data = np.array(label_data)
    # label_data = label_data.reshape(label_data.shape[1], label_data.shape[0])
    struct = t-1
    print('',struct)


    struct_record = struct
    length_data = len(data)
    length_struct = len(struct)
    diff_array = np.array([i for i in range(0,data.shape[0],1)])
    t_U = np.setdiff1d(diff_array, t-1)
    label_U.append(label_train[t_U])
    label_U = np.array(label_U)
    label_U = label_U.reshape(label_U.shape[1], label_U.shape[0])

    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        for i in range(0, len(struct),1 ):
            data_neigh.append(nneigh[int(struct[i])])
        data_neigh = np.array(data_neigh)


        struct = np.setdiff1d(data_neigh, struct_record)

        length_struct = len(struct)


        for i in range(0, length_struct):
            struct_record = np.append(struct_record, struct[i])
        struct_record = struct_record.reshape(struct_record.shape[0],1)

        data_TR = data
        lable_TR = label_data
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        lable_TR = np.array(lable_TR, dtype=np.int64)
        lable_TR = lable_TR.squeeze()

        try:
            for x, y in zip(data_TR, lable_TR):
                tree.update(x, y)
            label_data = tree.predict(data_TR)
            #print('label_data1', label_data)
        except:
            continue

        # for x, y in zip(data_TR, lable_TR):
        #     tree.update(x, y)
        # label_data = tree.predict(data)


    struct = struct_record
    length_struct = len(struct)

    del data_neigh

    def find(condition):
        res = np.nonzero(condition)
        return res
    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        k = 0
        for i in range(0, len(struct), 1):
            number_neigh_test = np.where(nneigh == struct[i])[0]
            number_neigh = find(nneigh == struct[i])[0]
            number_neigh = np.array(number_neigh)
            length_neigh = len(number_neigh)
            if len(struct) > 0:
                for j in range(0, length_neigh):
                    data_neigh.append(number_neigh[j])
            k = k + length_neigh
        data_neigh = np.array(data_neigh)



        struct = np.setdiff1d(data_neigh, struct_record)
        length_struct = len(struct)

        struct_record = list(struct_record)
        for i in range(0, length_struct):
            struct_record.append(struct[i])
        struct_record = np.array(struct_record)

        data_TR = data
        lable_TR = label_data
        data = list(data)
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)
        lable_TR = np.array(lable_TR, dtype=np.int64)
        lable_TR = lable_TR.squeeze()

        try:
            for x, y in zip(data_TR, lable_TR):
                tree.update(x, y)
            label_data = tree.predict(data)
            #print('label_data2', label_data)
        except:
            continue


    data = pd.DataFrame(train)
    data = data.values
    data = data[:]
    try:
#        label_data = label_data.astype('int')
        #print('1',data)
        #print('2',label_data)
        for x, y in zip(data, label_data):
            tree.update(x, y)
    except:
        # print('3', train[:40])
        # print('4', label_train[:40])
        for x, y in zip(train[:40], label_train[:40]):
            tree.update(x, y)

    #print('train',train)
    predict_label_train = tree.predict(train)

    return predict_label_train

def SSC_DensityPeaks_SVC_ensemble_clu(train_1, label_train_1, train_2, label_train_2, t, nneigh_x, nneigh_z): # , test, label_test #, clf, n, decay_choice, contribute_error_rate

    train_x = train_1
    label_train_x = label_train_1

    train_z = train_2
    label_train_z = label_train_2


    train_3 = pd.DataFrame(train_1)
    title_x = list(train_3.columns.values)
    features_x = title_x[:]

    tree_x = Vfdt(features_x, delta=0.001, nmin=20, tau=0.5)

    train_4 = pd.DataFrame(train_2)
    title_z = list(train_4.columns.values)
    features_z = title_z[:]

    tree_z = Vfdt(features_z, delta=0.001, nmin=20, tau=0.5)

    data_x = []
    label_data_x = []
    label_U_x = []

    data_x.append(train_x[t-1,]) # t-1
    data_x = np.array(data_x)
    data_x = data_x.reshape(data_x.shape[1], data_x.shape[2])

    label_data_x.append(label_train_x[t-1,]) # t-1
    label_data_x = np.array(label_data_x)
    label_data_x = label_data_x.reshape(label_data_x.shape[1], label_data_x.shape[0])
    struct_x = t-1

    struct_record_x = struct_x
    length_data_x = len(data_x)
    length_struct_x = len(struct_x)
    diff_array_x = np.array([i for i in range(0, data_x.shape[0], 1)])
    t_U_x = np.setdiff1d(diff_array_x, t-1)
    if len(t_U_x) != 0:
        label_U_x.append(label_train_x[t_U_x])
        label_U_x = np.array(label_U_x)
        label_U_x = label_U_x.reshape(label_U_x.shape[1], label_U_x.shape[0])

    '''Select first to point to the next node'''
    data_neigh_x = []
    while (len(struct_x) > 0):
        data_neigh_x = list(data_neigh_x)
        for i in range(0, len(struct_x),1 ):
            data_neigh_x.append(nneigh_x[int(struct_x[i])])
        data_neigh_x = np.array(data_neigh_x)


        struct_x = np.setdiff1d(data_neigh_x, struct_record_x)

        length_struct_x = len(struct_x)


        for i in range(0, length_struct_x):
            struct_record_x = np.append(struct_record_x, struct_x[i])
        struct_record_x = struct_record_x.reshape(struct_record_x.shape[0], 1)
        #print(struct_record_x)

        data_TR_x = data_x
        lable_TR_x = label_data_x
        data_list_x = list(data_x)
        label_data_list_x = list(label_data_x)
        for j in range(0, length_struct_x):
            data_list_x.append(train_x[int(struct_x[j])])
            label_data_list_x.append(label_train_x[j])
        data_x = np.array(data_list_x)
        label_data_x = np.array(label_data_list_x)
        length_data_x = len(data_x)
        lable_TR_x = np.array(lable_TR_x, dtype=np.int64)
        lable_TR_x = lable_TR_x.squeeze()

        try:
            #print(data_TR_x)
            #print(lable_TR_x)
            for x, y in zip(data_TR_x, lable_TR_x):
                tree_x.update(x, y)

            label_data_x = tree_x.predict(data_x)
            #print('label_data_x1',label_data_x)
        except:
            continue

    struct_x = struct_record_x
    length_struct_x = len(struct_x)
    del data_neigh_x

    data_neigh_x = []
    while (len(struct_x) > 0):
        data_neigh_x = list(data_neigh_x)
        k_x = 0
        for i in range(0, len(struct_x), 1):
            number_neigh_x = np.where(nneigh_x == struct_x[i])[0]
            number_neigh_x = np.array(number_neigh_x)
            length_neigh_x = len(number_neigh_x)
            if len(struct_x) > 0:
                for j in range(0, length_neigh_x):
                    data_neigh_x.append(number_neigh_x[j])
            k_x = k_x + length_neigh_x
        data_neigh_x = np.array(data_neigh_x)


        struct_x = np.setdiff1d(data_neigh_x, struct_record_x)
        length_struct_x = len(struct_x)

        struct_record_x = list(struct_record_x)
        for i in range(0, length_struct_x):
            struct_record_x.append(struct_x[i])
        struct_record_x = np.array(struct_record_x)

        data_TR_x = data_x
        lable_TR_x = label_data_x
        data_x = list(data_x)
        data_list_x = list(data_x)
        label_data_list_x = list(label_data_x)
        for j in range(0, length_struct_x):
            data_list_x.append(train_x[int(struct_x[j])])
            label_data_list_x.append(label_train_x[j])
        data_x = np.array(data_list_x)
        label_data_x = np.array(label_data_list_x)
        length_data_x = len(data_x)  # 求数据长度
        lable_TR_x = np.array(lable_TR_x, dtype=np.int64)
        lable_TR_x = lable_TR_x.squeeze()
        try:
            for x, y in zip(data_TR_x, lable_TR_x):
                tree_x.update(x, y)
            label_data_x = tree_x.predict(data_x)
            # print('label_data_x2', label_data_x)
        except:
            continue

    data_z = []
    label_data_z = []
    label_U_z = []

    data_z.append(train_z[t-1,]) # t-1
    data_z = np.array(data_z)
    data_z = data_z.reshape(data_z.shape[1], data_z.shape[2])

    label_data_z.append(label_train_z[t-1,]) # t-1
    label_data_z = np.array(label_data_z)
    label_data_z = label_data_z.reshape(label_data_z.shape[1], label_data_z.shape[0])
    struct_z = t-1

    struct_record_z = struct_z
    length_data_z = len(data_z)
    length_struct_z = len(struct_z)
    diff_array_z = np.array([i for i in range(0, data_z.shape[0], 1)])
    t_U_z = np.setdiff1d(diff_array_z, t-1)
    if len(t_U_z) != 0:
        label_U_z.append(label_train_z[t_U_z])
        label_U_z = np.array(label_U_z)
        label_U_z = label_U_z.reshape(label_U_z.shape[1], label_U_z.shape[0])

    '''Select first to point to the next node'''
    data_neigh_z = []
    while (len(struct_z) > 0):
        data_neigh_z = list(data_neigh_z)
        for i in range(0, len(struct_z),1 ):
            data_neigh_z.append(nneigh_z[int(struct_z[i])])
        data_neigh_z = np.array(data_neigh_z)
        struct_z = np.setdiff1d(data_neigh_z, struct_record_z)
        length_struct_z = len(struct_z)

        for i in range(0, length_struct_z):
            struct_record_z = np.append(struct_record_z, struct_z[i])
        struct_record_z = struct_record_z.reshape(struct_record_z.shape[0], 1)

        data_TR_z = data_z
        lable_TR_z = label_data_z
        data_list_z = list(data_z)
        label_data_list_z = list(label_data_z)
        for j in range(0, length_struct_z):
            data_list_z.append(train_z[int(struct_z[j])])
            label_data_list_z.append(label_train_z[j])
        data_z = np.array(data_list_z)
        label_data_z = np.array(label_data_list_z)
        length_data_z = len(data_z)
        lable_TR_z = np.array(lable_TR_z, dtype=np.int64)
        lable_TR_z = lable_TR_z.squeeze()
        try:
            for x, y in zip(data_TR_z, lable_TR_z):
                tree_z.update(x, y)
            label_data_z = tree_x.predict(data_z)
            #print('label_data_z1', label_data_z)
        except:
            continue

    struct_z = struct_record_z
    length_struct_z = len(struct_z)
    del data_neigh_z

    data_neigh_z = []
    while (len(struct_z) > 0):
        data_neigh_z = list(data_neigh_z)
        k_z = 0
        for i in range(0, len(struct_z), 1):
            number_neigh_z = np.where(nneigh_z == struct_z[i])[0]
            number_neigh_z = np.array(number_neigh_z)
            length_neigh_z = len(number_neigh_z)
            if len(struct_z) > 0:
                for j in range(0, length_neigh_z):
                    data_neigh_z.append(number_neigh_z[j])
            k_z = k_z + length_neigh_z
        data_neigh_z = np.array(data_neigh_z)

        struct_z = np.setdiff1d(data_neigh_z, struct_record_z)
        length_struct_z = len(struct_z)

        struct_record_z = list(struct_record_z)
        for i in range(0, length_struct_z):
            struct_record_z.append(struct_z[i])
        struct_record_z = np.array(struct_record_z)

        data_TR_z = data_z
        lable_TR_z = label_data_z
        data_z = list(data_z)
        data_list_z = list(data_z)
        label_data_list_z = list(label_data_z)
        for j in range(0, length_struct_z):
            data_list_z.append(train_z[int(struct_z[j])])
            label_data_list_z.append(label_train_z[j])
        data_z = np.array(data_list_z)
        label_data_z = np.array(label_data_list_z)
        length_data_z = len(data_z)
        lable_TR_z = np.array(lable_TR_z, dtype=np.int64)
        lable_TR_z = lable_TR_z.squeeze()
        try:
            for x, y in zip(data_TR_z, lable_TR_z):
                tree_z.update(x, y)
            label_data_z = tree_x.predict(data_z)
            #print('label_data_z2', label_data_z)
        except:
            continue

    '''
        Ensemble Learning
    '''
    # label_data_x = label_data_x.astype('int')
    # label_data_z = label_data_z.astype('int')
    if len(data_x) != 0:
        if len(set(label_data_x)) != 1:
            for x, y in zip(data_x, label_data_x):
                tree_x.update(x, y)
    if len(data_z) != 0:
        if len(set(label_data_z)) != 1:
            for x, y in zip(data_z, label_data_z):
                tree_z.update(x, y)

    def sigmoid(x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    lamda = 0.5
    predict_label = []
    length = len(train_1)
    # print("length", length)
    for n in range(length):
        train_1_1 = train_1[n, ]
        train_2_1 = train_2[n, ]
        try:
            predict_label_train_1 = tree_x._predict_proba_lr(np.array(train_1_1).reshape(1, -1))
            predict_label_train_1 = np.argmax(predict_label_train_1)
        except IndexError:
            predict_label_train_1 = np.random.randint(0, 2, 1)
        except :
            predict_label_train_1 = np.random.randint(0, 2, 1)

        try:
            predict_label_train_2 = tree_z._predict_proba_lr(np.array(train_2_1).reshape(1, -1))
            predict_label_train_2 = np.argmax(predict_label_train_2)
        except IndexError:
            predict_label_train_2 = np.random.randint(0, 2, 1)
        except :
            predict_label_train_2 = np.random.randint(0, 2, 1)

        predict_label_train = sigmoid(lamda * predict_label_train_1 + (1.0 - lamda) * predict_label_train_2)
        if predict_label_train > 0.5:
            predict_label_train = 1
        else:
            predict_label_train = 0
        predict_label.append(predict_label_train)

    predict_label = np.array(predict_label)
    predict_label = predict_label.reshape(predict_label.shape[0])
    return predict_label


def SSC_DensityPeaks_SVC_clu_test(train, label_train, t,
                             nneigh):  # , test, label_test #, clf, n, decay_choice, contribute_error_rate

    data = []
    label_data = []
    label_U = []

    train_x = pd.DataFrame(train)
    title = list(train_x.columns.values)
    features = title[:]
    tree = Vfdt(features, delta=0.001, nmin=30, tau=0.5)

    # print(t)
    data.append(train[t - 1,])  # t-1
    data = np.array(data)
    data = data.reshape(data.shape[1], data.shape[2])

    label_data.append(label_train[t - 1,])  # t-1
    label_data = np.array(label_data)
    # label_data = label_data.reshape(label_data.shape[1], label_data.shape[0])
    struct = t - 1

    struct_record = struct
    length_data = len(data)
    length_struct = len(struct)
    diff_array = np.array([i for i in range(0, data.shape[0], 1)])
    t_U = np.setdiff1d(diff_array, t - 1)
    label_U.append(label_train[t_U])
    label_U = np.array(label_U)
    label_U = label_U.reshape(label_U.shape[1], label_U.shape[0])

    data_neigh = []
    while (len(struct) > 0):
        data_neigh = list(data_neigh)
        for i in range(0, len(struct), 1):
            data_neigh.append(nneigh[int(struct[i])])
        data_neigh = np.array(data_neigh)

        struct = np.setdiff1d(data_neigh, struct_record)

        length_struct = len(struct)

        for i in range(0, length_struct):
            struct_record = np.append(struct_record, struct[i])
        struct_record = struct_record.reshape(struct_record.shape[0], 1)

        data_TR = data
        lable_TR = label_data
        data_list = list(data)
        label_data_list = list(label_data)
        for j in range(0, length_struct):
            data_list.append(train[int(struct[j])])
            label_data_list.append(label_train[j])
        data = np.array(data_list)
        label_data = np.array(label_data_list)
        length_data = len(data)

        lable_TR = np.array(lable_TR, dtype=np.int64)
        lable_TR = lable_TR.squeeze()
        #print('lable_TR',lable_TR)
        for x, y in zip(data_TR, lable_TR):
            # print('x',x)
            # print('y',y)
            tree.update(x, y)
        print(data_TR.shape)
        print(lable_TR.shape)

        label_data = tree.predict(data_TR)
        #tree.print_tree(tree.root)
        print('label_data1', label_data)

        return label_data