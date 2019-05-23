# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/5/19

'''
将数据改为每一个词一行的格式
'''

dir_path = "../data/trainset/train_cws.txt"
with open(dir_path,"r+",encoding="utf-8")as f:
    lines = f.readlines()

for line in lines:
    print(line.split(" "))