'''
 detect the jump point to determine the number of cluster automatically
'''
from scipy.stats import betaprime
from scipy.stats import pareto
import numpy as np
import pandas as pd
import math
import sys
import copy


class Change_point():
    def __init__(self, gamma=1 / 2, sensitivity=0.05):
        self.dynamic_window = []
        self.gamma = gamma
        self.sensitivity = sensitivity
        self.cushion = None

    def insert_into_window(self, value):
        self.dynamic_window.append(value)

    def get_dynamic_window(self):
        return self.dynamic_window

    def shrink_window(self, position):
        for i in range(position + 1):
            self.dynamic_window.remove(self.dynamic_window[0])

    def shrink_list(self, list, position):
        for i in range(position + 1):
            list.remove(list[0])

    def detect_change(self):
        estimated_change_point = -1
        N = len(self.dynamic_window)
        self.cushion = max(100, int(math.floor(math.pow(N, self.gamma))))
        threshold = -math.log(self.sensitivity)

        w = 0
        k_at_max_w = -1
        start = N // 2
        end = N - 3
        for k in range(start, end + 1):
            if self.calculateMean(0, k - 1) <= 0.95 * self.calculateMean(k, N - 1):
                skn = 0
                scalePreChange = self.calParetoDistScale(0, k - 1)
                shapePreChange = self.calculateParetoDistShape(scalePreChange, 0, k - 1)

                scalePostChange = self.calParetoDistScale(k, N - 1)
                shapePostChange = self.calculateParetoDistShape(scalePostChange, k, N - 1)

                preParetoDist = betaprime(1, shapePreChange, scale=scalePreChange)
                postParetoDist = betaprime(1, shapePostChange, scale=scalePostChange)
                # preParetoDist = pareto(b=shapePreChange, loc=1, scale=scalePreChange)
                # postParetoDist = pareto(b=shapePostChange, loc=1, scale=scalePostChange)

                maxPr = sys.float_info.min
                minPr = sys.float_info.max
                maxPo = sys.float_info.min
                minPo = sys.float_info.max

                for i in range(k, N):
                    try:
                        postValue = postParetoDist.logpdf(self.dynamic_window[i])
                        preValue = preParetoDist.logpdf(self.dynamic_window[i])
                        if postValue == 0 or preValue == 0:
                            break
                        if postValue > maxPo:
                            maxPo = postValue
                        if postValue < minPo:
                            minPo = postValue
                        if preValue > maxPr:
                            maxPr = preValue
                        if preValue < minPr:
                            minPr = preValue
                    except:

                        skn = 0
                        break

                for i in range(k, N):
                    try:
                        postValue = postParetoDist.logpdf(self.dynamic_window[i])
                        preValue = preParetoDist.logpdf(self.dynamic_window[i])
                        pr = (postValue - minPo) / (maxPo - minPo)
                        po = (preValue - minPr) / (maxPr - minPr)

                        if po == 0 or pr == 0:
                            break
                        skn += math.log(po / pr)
                        # skn += math.log(postValue / preValue)
                    except:
                        # print("error occurred")
                        skn = 0
                        break

                if skn > w:
                    w = skn
                    k_at_max_w = k

        if w >= threshold / N * 2 and k_at_max_w != -1:
            estimated_change_point = k_at_max_w

        return estimated_change_point

    def calParetoDistScale(self, begin, to):
        minV = self.findMin(begin, to)
        if minV == 0:
            minV = minV + 0.00001
        return minV

    def findMin(self, begin, to):
        minValue = sys.float_info.max
        length = to - begin + 1
        for i in range(length):
            if self.dynamic_window[i] < minValue:
                minValue = self.dynamic_window[i]

        return minValue

    def calculateParetoDistShape(self, Scale, begin, to):
        sum = 0
        constV = math.log(Scale)
        length = to - begin + 1
        # print('self.dynamic_window:,',self.dynamic_window)
        for i in range(length):
            # if self.dynamic_window[i] == 0.0 or self.dynamic_window[i] == -0.0:
            #     continue
            # else:
            #     sum += math.log(self.dynamic_window[i]) - constV
            if self.dynamic_window[i] == 0.0 or self.dynamic_window[i] == -0.0:
                tr = self.dynamic_window[i]
                new_tr = tr + 0.00001
                if new_tr <= 0:
                    new_tr = 0.0001
                sum += math.log(new_tr) - constV
            else:
                sum += math.log(self.dynamic_window[i]) - constV
        if sum == 0:
            sum = -constV
        return length / sum

    """
    calculate mean of the elements in dynamicWindow
    both of the indices from and to are inclusive
    """

    def calculateMean(self, begin, to):
        sum = 0.0
        for i in range(begin, to + 1):
            sum += self.dynamic_window[i]

        return sum / (to - begin + 1)

    def calculateListMean(self, list_, begin, to):
        sum = 0.0
        for i in range(begin, to + 1):
            sum += list_[i]

        return sum / (to - begin + 1)

    def calculateVariance(self, begin, to):
        sumOfSquares = 0.0
        mean = self.calculateMean(begin, to)
        for i in range(begin, to):
            sumOfSquares += (self.dynamic_window[i] - mean) * (self.dynamic_window[i] - mean)

        return sumOfSquares / (to - begin + 1)

    def num_cluster(self, rho_multi_delta, n=30):  # type(rho_multi_delta)
        rhodelta = copy.deepcopy(rho_multi_delta)
        value = rhodelta[0:n]
        value = list(value)

        value.reverse()
        for i in range(len(value)):
            self.insert_into_window(value[i])

        changepoint = self.detect_change()
        if changepoint != -1:
            number_cluster = int(n - 1 - changepoint)
        else:
            number_cluster = 1

        return number_cluster


def main(path):
    df = pd.read_csv(path, header=None, sep=',')
    df = df.values[0]
    rhodelta = list(df)
    rhodelta = rhodelta[:20]
    rhodelta.reverse()
    print(len(rhodelta))
    value = []
    for i in range(len(rhodelta)):
        value.append(rhodelta[i])
    PareChangePoint = Change_point((1 / 2), 0.05)
    for i in range(len(value)):
        PareChangePoint.insert_into_window(value[i])
        print(value[i])

    changePoint = PareChangePoint.detect_change()

    return changePoint

if __name__ == '__main__':
    path = 'E:/xxdm/sscadpsrc-master/AU/Data Sets/au2_10000.csv'
    changePoint = main(path)
    print('changePointIndex', changePoint)
    list_ = [1, 0.9408793, 0.81069136, 0.7185438, 0.52959245, 0.5113219, 0.4837961, 0.47286355, 0.4712481, 0.45249987,
             0.44741607, 0.44439897, 0.43029112, 0.4233997, 0.40283474, 0.3919778, 0.38501608, 0.38444453, 0.38282862,
             0.37915555, 0.36572105, 0.3631272, 0.3606204, 0.35803863, 0.3565097, 0.3546012, 0.35113776, 0.35077932,
             0.3504002, 0.34896547]
    list_2 = [1, 0.99842525, 0.34642074, 0.28553417, 0.20135657, 0.15913044, 0.14404193, 0.14271843, 0.13992552,
              0.13751327, 0.13490897, 0.13308355, 0.13165511, 0.12969053, 0.1280579, 0.12783092, 0.12643377, 0.12083419,
              0.120462835, 0.11915514, 0.11665378, 0.11570541, 0.11550522, 0.11536463, 0.11522858, 0.115130626,
              0.112640016, 0.11242511, 0.11167092, 0.111236475]
    change_point = Change_point()
    num_cluster1 = change_point.num_cluster(list_)

    change_point2 = Change_point()
    num_cluster2 = change_point2.num_cluster(list_2)
    print(num_cluster1, num_cluster2)