import random
import pandas as pd
import numpy as np
from collections import defaultdict, Counter, deque #deque类为双向队列
import copy
# N = 200
# k = 5
# data = 'mushroom.training'
class Kmodes:
    def __init__(self, k_value, center):
        self.k_value = k_value  #初始化的聚簇数量
        self.center = center    #指定的聚簇中心，初始化的格式为 ndarray

    def continuous_convert_discrete(self, data): #data数据必须是[[..], [..],...,[..]] liu
        new_data = copy.deepcopy(data)
        convert_data = []  #保存转换过的数据
        num_attribute = len(new_data[0]) #属性的个数
        for i in range(num_attribute): #依次处理每个属性
            attr = [example[i] for example in new_data] #某个属性的所有取值
            unique_value = len(np.unique(attr))
            if isinstance(attr[0], str): #如果某个属性为离散属性则不做处理
                continue
            if unique_value <= 10:       #当某个属性所有不同的取值在10个之内时 如[0,1,2] 也不做处理
                continue

            #连续属性则离散化
            attr.sort() #将连续属性从小到大排序
            sorted_intervals = [] #初始化一个空的列表用于保存划分区间信息
            vals = int(len(attr)/10)
            for j in range(0, len(attr),vals): #将属性划分为10个相等的区间，注意这里先是每按200个样本做一次处理后面可能会改动
                c = attr[j:j+vals]
                sorted_intervals.append(c)

            for k in range(len(new_data)): #遍历每个样本将对应的连续属性值改为相应的区间索引
                for k_ in range(len(sorted_intervals)): #遍历索引区间找到对应的索引
                    #条件为大于等于区间第一个值，小于等于区间的最后一个值
                    if new_data[k][i] >= sorted_intervals[k_][0] and new_data[k][i] <= sorted_intervals[k_][-1]:
                        new_data[k][i] = k_

            for z in range(len(self.center)):#遍历每个初始化的聚簇中心将对应的连续属性值改为相应的区间索引
                for z_ in range(len(sorted_intervals)):
                    if self.center[z][i] >= sorted_intervals[z_][0] and self.center[z][i] <= sorted_intervals[z_][-1]:
                        self.center[z][i] = z_

        return new_data


    def is_converged(self, centroids, old_centroids):
        return set([tuple(a) for a in centroids]) == set([tuple(b) for b in old_centroids])
        # 当聚类中心不再变化时，判断为收敛

    def get_distance(self, x, c):
        return np.sum(np.array(x) != np.array(c), axis = 0) #判断两个ndarray数组对应不相等的元素个数

    def get_clusters(self, X, centroids):
        clusters = defaultdict(list) #初始化一个有默认值的空字典，如{'f':[]}
        for x in X: #为每个样本分配至距离最小的簇中
            # cluster is a num to indicate the # of centroids
            cluster = np.argsort([self.get_distance(x, c) for c in centroids])[0] #cluster为距离最小的簇中心的索引
            clusters[cluster].append(x)
        return clusters #类型为字典

    def get_centeroids(self, old_centroids, clusters): #聚类之后重新分配簇中心
        new_centroids = []
        keys = sorted(clusters.keys()) #此时的keys是一个list
        for k in keys: #依次遍历每个簇
            points = np.array(clusters[k]) #获得簇中所有样本，为ndarray类型[[1,2,1],[0,1,0],...,[1,1,2]]
            mode = [Counter(points[:, i]).most_common(1)[0][0] for i in range(len(old_centroids[0]))]
            #mode 返回每个属性出现最多的元素
            # mode.append('PAD')
            new_centroids.append(mode)
        return new_centroids

    def find_centers(self, X):
        # old_centroids = random.sample(X, K)
        # centroids = random.sample(list(X), self.k_value)
        old_centroids = random.sample(list(X), self.k_value) #从所有样本中随机选k个作为初始化的旧的簇中心
        centroids = self.center             #指定的聚簇中心
        clusters = {}
        iteration = 0                       #迭代次数

        while not self.is_converged(centroids, old_centroids): #while not x 当not x为True时循环，即当x为false时循环
            old_centroids = centroids                     #当聚类中心没有收敛时执行循环
            clusters = self.get_clusters(X, centroids)    #用较新的聚簇中心聚类，并返回字典类型的的聚簇集合
            centroids = self.get_centeroids(old_centroids, clusters) #一轮新的聚类结果产生后在每个簇中重新分配簇中心
            iteration += 1
        return centroids, clusters

    def get_labels(self, clusters, new_data):
        new_data = np.array(new_data)
        labels_list = []
        for i in range(len(new_data)):#遍历每个样本，记录对应的聚类结果
            for key, values in clusters.items():
                for value in np.array(values):
                    if all(new_data[i] == value): #all()方法比较两个ndarray是否完全相等
                        labels_list.append(key)
                        break

        return np.array(labels_list) #最重要的结果：返回每个样本使用Kmodes算法得到的类别标签 类型为list

    def find_intervals_info(self, data, centroids): #记录每个聚簇中心离散化后的索引值对应的区间 data为原始样本 liu
        intervals_info = copy.deepcopy(centroids) #复制聚簇中心的信息
        new_data = copy.deepcopy(data)
        num_attribute = len(new_data[0])  #属性的个数
        for i in range(num_attribute):  #依次处理每个属性 可定位至每个连续属性
            attr = [example[i] for example in new_data]  #某个属性的所有取值
            unique_value = len(np.unique(attr)) #所有取值中不重复的个数
            if isinstance(attr[0], str):  # 如果某个属性为离散属性则不做处理
                continue
            if unique_value <= 10:  # 当某个属性所有不同的取值在10个之内时 如[0,1,2] 也不做处理
                continue

            # 连续属性则离散化
            attr.sort()  # 将连续属性从小到大排序
            sorted_intervals = [] #初始化一个空的列表用于保存划分区间信息
            vals = int(len(attr) / 10)
            for j in range(0, len(attr), vals):  # 将属性划分为10个相等的区间，注意这里先是每按200个样本做一次处理后面可能会改动
                c = attr[j:j + vals]
                sorted_intervals.append(c)

            for z in range(len(intervals_info)): #找到满足条件的连续属性后，把每个簇对应的连续属性索引改为区间
                intervals_info[z][i] = sorted_intervals[int(intervals_info[z][i])]

        return intervals_info

    def get_purity(self, clusters, centroids, num_instances): #纯度，分到同一个簇中的最多类别的数量占样本总量的比例
        counts = 0
        for k in clusters.keys(): #遍历每个簇
            labels = np.array(clusters[k])[:, -1] #记录每个簇中所有样本的标签信息
            counts += Counter(labels).most_common(1)[0][1] #most_common(n)返回Counter结果中的前n个结果
                                                           #Counter为计数类，返回对应元素及出现的次数，从大到小排列
        return float(counts)/num_instances                 #第一个为出现次数最多的元素和它对应出现的次数

def main():
    # X = [x1, x2, ..., label]
    # X = get_data(data)     #X为list类型
    # num_instances = len(X) #样本的数量
    # centroids, clusters, iteration= find_centers(X, 10)
    # purity = get_purity(clusters, centroids, num_instances)
    # # print centroids, iteration
    # for k in clusters.keys():
    #         points = np.array(clusters[k])
    #         class_attr = Counter(points[:, -1]).most_common(1)
    #         print (class_attr)
    # print ('\n')
    # print('The purity for the task is %f' % purity)
    df = pd.read_csv('bank200.csv', header=0, sep=';')
    X = df.values  # 获得数据 ndarray形式
    print(X)
    num_instances = len(X)  # 样本的数量
    print('num_instances:', num_instances)
    clf = Kmodes(k_value=2)  # 初始化一个Kmodes类
    new_data = clf.continuous_convert_discrete(X)
    print(new_data, type(new_data))

    print('--------------------------------------------------------------------------------')
    centroids, clusters = clf.find_centers(new_data)
    print('centroids:',centroids)
    # print('clusters:',clusters)

    intervals_info = clf.find_intervals_info(X, centroids)
    print('intervals_info:', intervals_info)
    print(len(intervals_info), type(intervals_info[0][0][0]))
    print('--------------------------------------------------------------------------------------')

    purity = clf.get_purity(clusters, centroids, num_instances)
    print(purity)
    for k in clusters.keys():
        points = np.array(clusters[k])
        class_attr = Counter(points[:, -1]).most_common(1)
        print(class_attr)
    label_list = clf.get_labels(clusters, list(new_data))
    print('label_list:', label_list)


if __name__ == '__main__':
    main()










