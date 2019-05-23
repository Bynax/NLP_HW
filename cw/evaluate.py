# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/5/19
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

dev_path = "../data/devset/val_cws.txt"
para_path = "./para"


def cal_precision(raw_path, test_path):
    """
    :param raw_path: 标准文本
    :param test_path: 使用模型生成的文本
    :return:返回准确率
    """
    # 切分中正确分词数/切分结果中所有分词数


    pass


def cal_recall(raw_path, test_path):
    """
    :param raw_path: 标准文本
    :param test_path: 使用模型生成的文本
    :return:返回准确率
    """
    #切分结果中正确分词数/标准答案中所有分词数

    pass


def cal_f1(precision, recall):
    """
    :param precision: 模型准确率
    :param recall: 模型召回率
    :return:返回f1
    """

    pass

if __name__ == '__main__':
    with open(dev_path,"r",encoding="utf-8")as f:
        a = f.read()
    print(a)
    print(a.split(" "))