# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/5/19

from cw.dataset import Preprocess as p

dev_path = "../data/devset/val_cws.txt"
para_path = "./para"


class Evaluate(object):
    """
    度量模型，包括计算accruary、recall and F1_score
    """
    @classmethod
    def __cal_accurary(cls, reference_list, predict_list):
        """
        :param reference_list: 参考列表
        :param predict_list: 预测列表
        :return:返回准确率
        """
        # 切分中正确分词数/切分结果中所有分词数
        total_ref_num = len(predict_list)

        correct_words = cls.__correct_words(reference_list, predict_list)

        return correct_words / total_ref_num

    @classmethod
    def __cal_recall(cls, reference_list, predict_list):
        """
        :param reference_list: 参考列表
        :param predict_list: 预测列表
        :return:返回召回率
        """
        # 切分结果中正确分词数/标准答案中所有分词数

        total_ref_num = len(reference_list)

        correct_words = cls.__correct_words(reference_list, predict_list)

        return correct_words / total_ref_num

    @staticmethod
    def __cal_f1(accuracy, recall):
        """
        :param accuracy: 模型准确率
        :param recall: 模型召回率
        :return:返回f1
        """
        return 2 * (accuracy * recall) / (accuracy + recall)

    @staticmethod
    def __correct_words(reference_list, predict_list):
        """
        :param reference_list: 参考列表
        :param predict_list: 预测列表
        :return: 正确预测的数目
        """
        # 调用的时候 assert两个字符串长度相等
        correct_num = 0
        index_predict = 0
        index_reference = 0
        index_lreference = 0
        index_lpredict = 0
        len_lreference = len(reference_list)
        len_lpredict = len(predict_list)
        while index_lreference < len_lreference and index_lpredict < len_lpredict:
            if predict_list[index_lpredict] == reference_list[index_lreference]:  # match
                correct_num += 1
                index_predict += len(predict_list[index_lpredict])
                index_reference += len(reference_list[index_lreference])
                index_lpredict += 1
                index_lreference += 1
            else:
                if index_predict == index_reference:
                    index_predict += len(predict_list[index_lpredict])  # move
                    index_reference += len(reference_list[index_lreference])
                    index_lpredict += 1
                    index_lreference += 1
                elif index_predict < index_reference:
                    index_predict += len(predict_list[index_lpredict])
                    index_lpredict += 1
                elif index_predict > index_reference:
                    index_reference += len(reference_list[index_lreference])  # move
                    index_lreference += 1
        return correct_num

    @classmethod
    def evaluate(cls, reference, predict):
        """
        :param reference: 参考文本
        :param predict: 模型预测文本
        :return: 模型的accuracy/recall/F1_score
        """
        print(len(reference.replace(" ", "")))
        print(len(predict.replace(" ", "")))
        assert len(reference.replace(" ", "")) == len(predict.replace(" ", ""))  # 要比较的字符串除空格外字数要相等
        reference_list = reference.split(" ")
        predict_list = predict.split(" ")
        accuracy = cls.__cal_accurary(reference_list, predict_list)
        recall = cls.__cal_recall(reference_list, predict_list)
        f1_score = cls.__cal_f1(accuracy, recall)
        return accuracy, recall, f1_score


if __name__ == '__main__':
    lines = p.read_text_file("../data/trainset/test.txt")

    # 删除空行
    lines = p.del_blank_lines(lines)
    for line in lines:
        print(line)
    # 删除中英文标点
    for index, line in enumerate(lines):
        result = p.del_num_word(p.del_special_symbol(p.del_english_word(line)))
        lines[index] = result

    for line in lines:
        print(line)
    lines.pop()

