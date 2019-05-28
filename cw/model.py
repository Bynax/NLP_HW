# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/5/19

import math
import os


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
        pi = [0] * 4  # npi[i]：i状态的个数
        a = [[0] * 4 for x in range(4)]  # na[i][j]：从i状态到j状态的转移个数
        b = [[0] * 65536 for x in range(4)]  # nb[i][o]：从i状态到o字符的个数
        # with open(train_path, "r", encoding="utf-8")as f:
        #     data = f.read()
        # tokens = data.split(' ')

        # 开始训练
        last_q = 2
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
