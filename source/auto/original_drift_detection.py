import numpy as np
import pandas as pd

class Drift_detection:
    def calcu_r_single_cluster(self, members, center): #计算单个概念簇的r 此时的样本和簇中心都是将连续属性离散化后的
        total_members = len(members) #一个簇中的样本总数
        r = 0
        attr_number = len(center) #属性个数
        for i in range(total_members): #遍历簇中的每一个样本
            temp = 0
            for j in range(attr_number): #遍历样本中的每一维属性
                if isinstance(members[i][j],str): #如果该维属性为离散属性
                    if members[i][j] == center[j]:
                        temp += 1
                else: #该维属性为连续属性，比较的是索引
                    if members[i][j] == center[j]:
                        temp += 1
            r_temp = temp / attr_number #得到一个样本到簇中心的r
            r = r + r_temp #累积计算过的r之和

        # 如果一个簇的样本越相似，在空间中体现为越紧密，计算出来的簇的半径越大
        return  r / total_members


    def calcu_r(self, cluster_info, center_list): #计算一个概念簇集的半径 可同时用于计算r_hist,r_new
        cluster_num = len(center_list) #概念簇集中的概念簇个数
        r = 0
        for key,value in cluster_info.items(): #cluster_info用于保存簇类信息，类型为字典{第0个簇：[簇中所有点的坐标],...}
            try:
                r_temp = self.calcu_r_single_cluster(value, center_list[key])
                r = r + r_temp
            except IndexError:
                continue
        return r / cluster_num

    #intervals_hist字典类型，保存着每个簇中心离散化后连续属性对应的区间信息{0:[[1.2,2.3],'red',[6.8,9.0]],1:[[2.3,3.4],'green',[8.7,8.9]]}
    def calcu_dist(self, center_hist, center_new, intervals_hist, intevals_new): #计算两个概念簇集之间的距离 考虑的情况是hist和new中的簇数量一致
        attr_num = len(center_new[0]) #属性个数
        max_sim_list = [] #用于保存每个概念簇的最大相似度
        print('center_hist:',center_hist)
        print('center_new:',center_new)
        for i in range(len(center_new)): #遍历new中的概念簇
            sim = 0 #初始化相似度为0
            max_sim = -1
            for j in range(len(center_hist)): #在hist概念簇中找到最小的dist即最大的相似度
                temp_sim = 0 #初始化两个簇之间各个属性相似度之和
                for z in range(attr_num): #遍历比较每一维度属性
                    if isinstance(center_new[i][z],str) and center_new[i][z] ==center_hist[j][z]:#离散属性
                        temp_sim += 1 #离散属性相等 对应属性相似度+1
                    else: #连续属性
                        interval_list_new = intevals_new[i][z] #定位到要比较的区间
                        interval_list_hist = intervals_hist[j][z]
                        # print('intervals_hist',intervals_hist)
                        # print(type(intervals_hist))
                        # print('intervals_hist',intervals_hist.key())
                        # print('intevals_new',intevals_new)
                        # print('interval_list_hist',interval_list_hist)
                        # print('interval_list_new',interval_list_new)
                        if isinstance(interval_list_new, float) or isinstance(interval_list_new, int): #属性取值1或0
                            if interval_list_new == interval_list_hist:
                                temp_sim += 1
                        else:
                            len_new = len(interval_list_new)
                            len_hist = len(interval_list_hist)
                            sim_1 = 0
                            sim_2 = 0
                            for i_ in range(len_new): #新簇区间在旧簇区间中的交集
                                if interval_list_new[i_] >= interval_list_hist[0] and interval_list_new[i_] <= interval_list_hist[-1]:
                                    sim_1 += 1
                            for j_ in range(len_hist): #旧簇区间在新簇区间中的交集
                                if interval_list_hist[j_] >= interval_list_new[0] and interval_list_hist[j_] <= interval_list_new[-1]:
                                    sim_2 += 1
                            temp_sim_con = (sim_1 + sim_2) / (len_new + len_hist) #连续属性的相似度[0,1]之间
                            temp_sim = temp_sim + temp_sim_con
                sim = temp_sim / attr_num #计算完一个历史概念簇集，除以属性个数，得到的相似度在[0,1]之间
                if sim > max_sim: #记录离历史概念簇中最大相似度
                    max_sim = sim
            max_sim_list.append(max_sim) #遍历一次center_hist得到一个新概念簇到历史概念簇集的最大相似度

        dist = 1 - (sum(max_sim_list) / len(max_sim_list)) #两个概念簇集合之间的距离定义为1减去他们之间的最大相似度

        return dist

    def calcu_dist_weight(self,center_hist, center_new,
                          cluster_info_hist,cluster_info_new,
                          intervals_hist, intevals_new): #将原论文中计算距离的方法改为加权平均
        attr_num = len(center_new[0])  # 属性个数
        max_sim_list = []  # 用于保存每个概念簇的最大相似度
        related_weight_list = [] #用于保存每个概念簇的最大相似度对应的权重
        num_data = 0
        for c in cluster_info_new.values():
            num_data = num_data + len(c)
        for d in cluster_info_hist.values():
            num_data = num_data + len(d)
        #此时获得了新旧概念簇集的所有样本总数

        # print('center_hist:', center_hist)
        # print('center_new:', center_new)
        for i in range(len(center_new)): #遍历new中的概念簇
            num_new = len(cluster_info_new[i]) #当前概念簇中的样本
            sim = 0 #初始化相似度为0
            weight = 0 #初始化权重为0
            max_sim = -1
            related_weight = -1
            for j in range(len(center_hist)): #在hist概念簇中找到最小的dist即最大的相似度
                num_hist = len(cluster_info_hist[j])
                temp_sim = 0 #初始化两个簇之间各个属性相似度之和
                for z in range(attr_num): #遍历比较每一维度属性
                    if isinstance(center_new[i][z],str) and center_new[i][z] ==center_hist[j][z]:#离散属性
                        temp_sim += 1 #离散属性相等 对应属性相似度+1
                    else: #连续属性
                        interval_list_new = intevals_new[i][z] #定位到要比较的区间
                        interval_list_hist = intervals_hist[j][z]
                        # print('intervals_hist',intervals_hist)
                        # print(type(intervals_hist))
                        # print('intervals_hist',intervals_hist.key())
                        # print('intevals_new',intevals_new)
                        # print('interval_list_hist',interval_list_hist)
                        # print('interval_list_new',interval_list_new)
                        if isinstance(interval_list_new, float) or isinstance(interval_list_new, int): #属性取值1或0
                            if interval_list_new == interval_list_hist:
                                temp_sim += 1
                        else:
                            len_new = len(interval_list_new)
                            len_hist = len(interval_list_hist)
                            sim_1 = 0
                            sim_2 = 0
                            for i_ in range(len_new): #新簇区间在旧簇区间中的交集
                                if interval_list_new[i_] >= interval_list_hist[0] and interval_list_new[i_] <= interval_list_hist[-1]:
                                    sim_1 += 1
                            for j_ in range(len_hist): #旧簇区间在新簇区间中的交集
                                if interval_list_hist[j_] >= interval_list_new[0] and interval_list_hist[j_] <= interval_list_new[-1]:
                                    sim_2 += 1
                            temp_sim_con = (sim_1 + sim_2) / (len_new + len_hist) #连续属性的相似度[0,1]之间
                            temp_sim = temp_sim + temp_sim_con
                sim = temp_sim / attr_num #计算完一个历史概念簇集，除以属性个数，得到的相似度在[0,1]之间
                weight = (num_new + num_hist) / num_data
                if sim > max_sim: #记录离历史概念簇中最大相似度及对应的权重
                    max_sim = sim
                    related_weight = weight

            max_sim_list.append(max_sim) #遍历一次center_hist得到一个新概念簇到历史概念簇集的最大相似度
            related_weight_list.append(related_weight) #遍历一次center_hist得到一个新概念簇到历史概念簇集的最大相似度对应的权重

        max_sim_list = np.array(max_sim_list) #概念簇集的最大相似度转换为ndarray型
        related_weight_list = np.array(related_weight_list)#概念簇集的最大相似度对应的权重
        final_sim = sum(max_sim_list*related_weight_list) #得到加权平均的概念簇集之间的相似度和

        dist = 1 - final_sim #得到最终的新旧概念簇集之间的距离
        # dist = 1 - (sum(max_sim_list) / len(max_sim_list)) #两个概念簇集合之间的距离定义为1减去他们之间的最大相似度
        return dist


    def judge_drift(self, dist, r_hist, r_new):
        flag = False
        if dist >= (r_hist + r_new):
            flag = True
        return flag










