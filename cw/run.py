# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/5/19
from cw.evaluate import Evaluate as evaluate
from cw.dataset import Preprocess as preprocess
from cw.model import Model as model
import os

train_set = "../data/trainset/train_cws.txt"  # 训练文本位置
dev_set = "../data/devset/val_cws.txt"  # 验证文本位置
test_set = "../data/testset1/test_cws1.txt"  # 开发文本位置
para_dir = "./para"  # 参数存放位置
result_dir = "./result"

if __name__ == '__main__':
    """-------------------train-----------------------"""
    # train_tokens = preprocess.read_text_file(train_set)
    # model.train(train_tokens, para_dir=para_dir)
    """-------------------segment-----------------------"""
    # pi_path = os.path.join(para_dir, "pi.txt")
    # a_path = os.path.join(para_dir, "A.txt")
    # b_path = os.path.join(para_dir, "B.txt")
    # result_path = os.path.join(result_dir,"result.txt")
    # dev_text = preprocess.read_text_file(dev_set).replace(" ","")
    # model.seg(dev_text,pi_path,a_path,b_path,result_path)

    """-------------------evaluate-----------------------"""
    result_path = os.path.join(result_dir,"result.txt")
    resut_text = preprocess.process_text(preprocess.read_text_file(result_path,True))
    dev_text = preprocess.process_text(preprocess.read_text_file(dev_set,True))
    resut_text = "".join(resut_text).replace("  "," ")
    dev_text = "".join(dev_text).replace("  "," ")
    print(resut_text)
    print(dev_text)
    print(evaluate.evaluate(dev_text,resut_text))





