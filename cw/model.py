# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/5/19

import math
import os
import numpy as np
from collections import Counter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



class Model(object):
    __infinite = float(-2 ** 31)

    @staticmethod
    def __log_normalize(a):
        s = 0
        for x in a:
            s += x
        if s == 0:
            print("Error..from log_normalize.")
            return
        s = math.log(s)
        for i in range(len(a)):
            if a[i] == 0:
                a[i] = Model.__infinite
            else:
                a[i] = math.log(a[i]) - s

    @classmethod
    def __mle(cls, tokens):  # 0B/1M/2E/3S
        """
        根据训练文本，计算hmm中的pi，A，B参数
        :param tokens:词列表
        :return:pi,a,b:对应hmm中所训练的三个参数
        """
        pi = np.zeros(4)  # npi[i]：i状态的个数
        a = np.zeros((4, 4))  # na[i][j]：从i状态到j状态的转移个数
        b = np.zeros((4, 65536))  # nb[i][o]：从i状态到o字符的个数
        # 开始训练
        last_q = 2  # 上一个状态 为计算A矩阵方便
        old_progress = 0
        print('进度：')
        for k, token in enumerate(tokens):
            progress = float(k) / float(len(tokens))
            if progress > old_progress + 0.1:
                print('%.3f%%' % (progress * 100))
                old_progress = progress
            token = token.strip()
            n = len(token)
            if n <= 0:
                continue
            if n == 1:
                pi[3] += 1
                a[last_q][3] += 1  # 上一个词的结束(last_q)到当前状态(3S)
                b[3][ord(token[0])] += 1
                last_q = 3
                continue
            # 初始向量
            pi[0] += 1
            pi[2] += 1
            pi[1] += (n - 2)
            # 转移矩阵
            a[last_q][0] += 1
            last_q = 2
            if n == 2:
                a[0][2] += 1
            else:
                a[0][1] += 1
                a[1][1] += (n - 3)
                a[1][2] += 1
            # 发射矩阵
            b[0][ord(token[0])] += 1
            b[2][ord(token[n - 1])] += 1
            for i in range(1, n - 1):
                b[1][ord(token[i])] += 1
        # b = b + 0.027
        b = cls.__smoothing(b)
        # 正则化
        cls.__log_normalize(pi)
        for i in range(4):
            cls.__log_normalize(a[i])
            cls.__log_normalize(b[i])
        return pi, a, b

    @staticmethod
    def __list_write(f, v):
        for a in v:
            f.write(str(a))
            f.write(' ')
        f.write('\n')

    @staticmethod
    def __save_parameter(pi, A, B, para_dir):
        pi_path = os.path.join(para_dir, "pi.txt")
        a_path = os.path.join(para_dir, "A.txt")
        b_path = os.path.join(para_dir, "B.txt")
        with open(pi_path, "w", encoding="utf-8")as f_pi, \
                open(a_path, "w", encoding="utf-8")as f_A, \
                open(b_path, "w", encoding="utf-8")as f_B:

            Model.__list_write(f_pi, pi)
            for a in A:
                Model.__list_write(f_A, a)
            for b in B:
                Model.__list_write(f_B, b)
        return pi_path, a_path, b_path

    @staticmethod
    def __load_train(pi_path, a_path, b_path):
        with open(pi_path, "r", encoding="utf-8")as p, open(a_path, "r", encoding="utf-8")as a, open(
                b_path, "r", encoding="utf-8")as b:
            lp = p.readlines()
            la = a.readlines()
            lb = b.readlines()
        for line in lp:
            pi = list(map(float, line.split(' ')[:-1]))

        A = [[] for x in range(4)]  # 转移矩阵：B/M/E/S
        i = 0
        for line in la:
            A[i] = list(map(float, line.split(' ')[:-1]))
            i += 1

        B = [[] for x in range(4)]
        i = 0
        for line in lb:
            B[i] = list(map(float, line.split(' ')[:-1]))
            i += 1
        return pi, A, B

    @staticmethod
    def __viterbi(pi, A, B, o):
        T = len(o)  # 观测序列
        delta = [[0 for i in range(4)] for t in range(T)]
        pre = [[0 for i in range(4)] for t in range(T)]  # 前一个状态   # pre[t][i]：t时刻的i状态，它的前一个状态是多少
        for i in range(4):
            delta[0][i] = pi[i] + B[i][ord(o[0])]
        for t in range(1, T):
            for i in range(4):
                delta[t][i] = delta[t - 1][0] + A[0][i]
                for j in range(1, 4):
                    vj = delta[t - 1][j] + A[j][i]
                    if delta[t][i] < vj:
                        delta[t][i] = vj
                        pre[t][i] = j
                delta[t][i] += B[i][ord(o[t])]
        decode = [-1 for t in range(T)]  # 解码：回溯查找最大路径
        q = 0
        for i in range(1, 4):
            if delta[T - 1][i] > delta[T - 1][q]:
                q = i
        decode[T - 1] = q
        for t in range(T - 2, -1, -1):
            q = pre[t + 1][q]
            decode[t] = q
        return decode

    @staticmethod
    def __segment(result_path, sentence, decode):
        N = len(sentence)
        i = 0
        with open(result_path, "a+", encoding="utf-8")as f:
            while i < N:  # B/M/E/S
                if decode[i] == 0 or decode[i] == 1:  # Begin
                    j = i + 1
                    while j < N:
                        if decode[j] == 2:
                            break
                        j += 1
                    f.write("{} ".format(sentence[i:j + 1]))
                    i = j + 1
                elif decode[i] == 3 or decode[i] == 2:  # single
                    f.write("{} ".format(sentence[i:i + 1]))
                    i += 1
                else:
                    print('Error:', i, decode[i])
                    i += 1

    @classmethod
    def train(cls, train_tokens, para_dir):
        """
        ：:param train_tokens: 词list
        ：:param para_dir:参数存放目录
        ：:return: pi,A,B三个参数存放的路径
        """
        pi, A, B = cls.__mle(tokens=train_tokens)
        return cls.__save_parameter(pi, A, B, para_dir)

    @classmethod
    def seg(cls, data, pi_path, a_path, b_path, result_path):
        """
        根据已有的训练参数对指定文本进行切分
        :param data: 要被切分的文本
        :param pi_path: 参数pi存放的文件位置
        :param a_path: 参数A存放的文件位置
        :param b_path: 参数B存放的文件位置
        :param result_path: 切分结果存放路径
        :return:
        """
        pi, A, B = cls.__load_train(pi_path, a_path, b_path)
        assert pi is not None and A is not None and B is not None  # ensure train has been exacute
        decode = cls.__viterbi(pi, A, B, data)
        cls.__segment(result_path, data, decode)

    @classmethod
    def __smoothing(cls, b):
        """
        将指定的B矩阵经过平滑后返回
        :param B: 待平滑的矩阵
        :return:平滑后的矩阵
        """
        # print("counter:{}".format(Counter(b.sum(axis=0)).most_common(20)))
        words_frequency = b.sum(axis=0)
        most_common40 = Counter(words_frequency).most_common()
        print(most_common40[-50])
        dict_most_common = {}
        for item in most_common40:
            dict_most_common[item[0]] = item[1]
        return cls.__good_turing(b, dict_most_common, 70, words_frequency)

    @staticmethod
    def __good_turing(b, dict_most_common, threshold, words_frequency):
        """
        使用good_turing平滑
        :param b: HMM中的B参数矩阵
        :param dict_most_common: 重新调整权重后得到的各个gram对应的频率，形如(2，6555536)表示出现2次的共有65536个gram
        :param threshold:表示需要调整的threshold 防止gap的出现
        :param words_frequency:字符频率列表，表示位置对应的unicode编码中汉字出现的频率
        :return: b:经过调整权重后的B矩阵
        """
        # 重新调整权重
        # for item in dict_most_common.items():
        #     print("{}:{}".format(item[0], item[1]))
        for i in range(threshold):
            dict_most_common[i] = (i + 1) * dict_most_common[(i + 1)] / dict_most_common[i]

        # print("-----"*20)
        # for item in dict_most_common.items():
        #     print("{}:{}".format(item[0], item[1]))

        # 重新调整b矩阵
        for word_index, word_frequency in enumerate(words_frequency):

            if word_frequency < threshold:  # 假设前面程序都正确，不会出现频率小于0的情况
                if word_frequency == 0:  # 频次为0的情况特殊，其他的处理逻辑相同
                    b[:, word_index] = dict_most_common[0] / 4
                else:
                    for i in range(4):
                        b[i, word_index] = dict_most_common[word_frequency] * (b[i, word_index] / word_frequency)
        return b


def fun(x, a, b):
    return a * (x ** b)


if __name__ == '__main__':
    a = [ (2.0, 203), (3.0, 127), (5.0, 90), (4.0, 82), (6.0, 79), (11.0, 49), (7.0, 48),
         (10.0, 46), (12.0, 45), (8.0, 43), (9.0, 39), (13.0, 37), (14.0, 36), (16.0, 31), (18.0, 26), (15.0, 25),
         (27.0, 22), (28.0, 21)]
    x = []
    y = []
    for aa in a:
        x.append(aa[0])
        y.append(aa[1])
    popt,_ = curve_fit(fun,x,y)
    plt.plot(x,y,"b-")
    y2 = [fun(i, popt[0],popt[1]) for i in x]
    plt.plot(x, y2, 'r--')
    plt.show()
    print(popt)


