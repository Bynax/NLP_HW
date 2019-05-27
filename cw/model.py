# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/5/19

import math
import os

infinite = float(-2 ** 31)
result_path = "./result/cw_result.txt"

def log_normalize(a):
    s = 0
    for x in a:
        s += x
    if s == 0:
        print("Error..from log_normalize.")
        return
    s = math.log(s)
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = infinite
        else:
            a[i] = math.log(a[i]) - s


def log_sum(a):
    if not a:  # a为空
        return infinite
    m = max(a)
    s = 0
    for t in a:
        s += math.exp(t - m)
    return m + math.log(s)


def calc_alpha(pi, A, B, o, alpha):
    for i in range(4):
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    T = len(o)
    temp = [0 for i in range(4)]
    del i
    for t in range(1, T):
        for i in range(4):
            for j in range(4):
                temp[j] = (alpha[t - 1][j] + A[j][i])
            alpha[t][i] = log_sum(temp)
            alpha[t][i] += B[i][ord(o[t])]


def calc_beta(pi, A, B, o, beta):
    T = len(o)
    for i in range(4):
        beta[T - 1][i] = 1
    temp = [0 for i in range(4)]
    del i
    for t in range(T - 2, -1, -1):
        for i in range(4):
            beta[t][i] = 0
            for j in range(4):
                temp[j] = A[i][j] + B[j][ord(o[t + 1])] + beta[t + 1][j]
            beta[t][i] += log_sum(temp)


def calc_gamma(alpha, beta, gamma):
    for t in range(len(alpha)):
        for i in range(4):
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(gamma[t])
        for i in range(4):
            gamma[t][i] -= s


def calc_ksi(alpha, beta, A, B, o, ksi):
    T = len(alpha)
    temp = [0 for x in range(16)]
    for t in range(T - 1):
        k = 0
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][ord(o[t + 1])] + beta[t + 1][j]
                temp[k] = ksi[t][i][j]
                k += 1
        s = log_sum(temp)
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -= s


def bw(pi, A, B, alpha, beta, gamma, ksi, o):
    T = len(alpha)
    for i in range(4):
        pi[i] = gamma[0][i]
    s1 = [0 for x in range(T - 1)]
    s2 = [0 for x in range(T - 1)]
    for i in range(4):
        for j in range(4):
            for t in range(T - 1):
                s1[t] = ksi[t][i][j]
                s2[t] = gamma[t][i]
            A[i][j] = log_sum(s1) - log_sum(s2)
    s1 = [0 for x in range(T)]
    s2 = [0 for x in range(T)]
    for i in range(4):
        print("bw", i)
        for k in range(65536):
            valid = 0
            if k % 10000 == 0:
                print("bw - k", k)
            for t in range(T):
                if ord(o[t]) == k:
                    s1[valid] = gamma[t][i]
                    valid += 1
                s2[t] = gamma[t][i]
            if valid == 0:
                B[i][k] = infinite
            else:
                B[i][k] = log_sum(s1[:valid]) - log_sum(s2)


# def baum_welch(pi, A, B):
#     sentence = f.read()[3:].decode('utf-8')
#     f.close()
#     T = len(sentence)
#     alpha = [[0 for i in range(4)] for t in range(T)]
#     beta = [[0 for i in range(4)] for t in range(T)]
#     gamma = [[0 for i in range(4)] for t in range(T)]
#     ksi = [[[0 for j in range(4)] for i in range(4)] for t in range(T-1)]
#     for time in range(3):
#         print ("calc_alpha")
#         calc_alpha(pi, A, B, sentence, alpha)    # alpha(t,i):给定lamda，在时刻t的状态为i且观测到o(1),o(2)...o(t)的概率
#         print ("calc_beta")
#         calc_beta(pi, A, B, sentence, beta)      # beta(t,i)：给定lamda和时刻t的状态i，观测到o(t+1),o(t+2)...oT的概率
#         print ("calc_gamma")
#         calc_gamma(alpha, beta, gamma)    # gamma(t,i)：给定lamda和O，在时刻t状态位于i的概率
#         print ("calc_ksi")
#         calc_ksi(alpha, beta, A, B, sentence, ksi)    # ksi(t,i,j)：给定lamda和O，在时刻t状态位于i且在时刻i+1，状态位于j的概率
#         print ("bw")
#         bw(pi, A, B, alpha, beta, gamma, ksi, sentence)
#         print ("time", time)
#         print ("Pi:", pi)
#         print ("A", A)


def mle(train_path):  # 0B/1M/2E/3S
    """
    根据训练文本，计算hmm中的pi，A，B参数
    :param train_path:训练文本所在位置
    :return:pi,a,b:对应hmm中所训练的三个参数
    """
    pi = [0] * 4  # npi[i]：i状态的个数
    a = [[0] * 4 for x in range(4)]  # na[i][j]：从i状态到j状态的转移个数
    b = [[0] * 65536 for x in range(4)]  # nb[i][o]：从i状态到o字符的个数
    with open(train_path, "r", encoding="utf-8")as f:
        data = f.read()
    tokens = data.split(' ')

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
    log_normalize(pi)
    for i in range(4):
        log_normalize(a[i])
        log_normalize(b[i])
    return pi, a, b


def list_write(f, v):
    for a in v:
        f.write(str(a))
        f.write(' ')
    f.write('\n')


def save_parameter(pi, A, B, para_dir):
    pi_path = os.path.join(para_dir, "pi.txt")
    a_path = os.path.join(para_dir, "A.txt")
    b_path = os.path.join(para_dir, "B.txt")
    with open(pi_path, "w", encoding="utf-8")as f_pi, \
            open(a_path, "w", encoding="utf-8")as f_A, \
            open(b_path, "w", encoding="utf-8")as f_B:

        list_write(f_pi, pi)
        for a in A:
            list_write(f_A, a)
        for b in B:
            list_write(f_B, b)
    return pi_path, a_path, b_path


def load_train(pi_path, a_path, b_path):
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


def viterbi(pi, A, B, o):
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


def segment(sentence, decode):
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
                print(sentence[i:j + 1], "|", )
                f.write("{} ".format(sentence[i:j + 1]))
                i = j + 1
            elif decode[i] == 3 or decode[i] == 2:  # single
                print(sentence[i:i + 1], "|", )
                f.write("{} ".format(sentence[i:i + 1]))
                i += 1
            else:
                print('Error:', i, decode[i])
                i += 1


def train(train_path, para_dir):
    """
    ：:param train_path:训练集文本位置
    ：:param para_dir:参数存放目录
    ：:return: pi,A,B三个参数存放的路径
    """
    pi, A, B = mle(train_path)
    return save_parameter(pi, A, B, para_dir)


def seg(data_path, pi_path, a_path, b_path):
    pi, A, B = load_train(pi_path, a_path, b_path)
    with open(data_path, "r", encoding="utf-8")as f:
        data = f.read()
    decode = viterbi(pi, A, B, data)
    segment(data, decode)

if __name__ == '__main__':
    train_path = "../data/trainset/test.txt"
    train(train_path,"./para")
