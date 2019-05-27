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

        correct_words = cls.__correct_words(reference_list, predict_list)

        total_ref_num = len(reference_list)
        return correct_words / total_ref_num

    @staticmethod
    def __cal_recall(true_list, predict_list):
        """
        :param reference_list: 参考列表
        :param predict_list: 预测列表
        :return:返回召回率
        """
        # 切分结果中正确分词数/标准答案中所有分词数

        pass

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
        while reference_list and predict_list:
            if predict_list[0] == reference_list[0]:  # match
                correct_num += 1
                index_predict += len(predict_list[0])
                index_reference += len(reference_list[0])
                predict_list.pop(0)
                reference_list.pop(0)
            else:
                if index_predict == index_reference:
                    index_predict += len(predict_list[0])  # move
                    index_reference += len(reference_list[0])
                    predict_list.pop(0)
                    reference_list.pop(0)
                elif index_predict < index_reference:
                    index_predict += len(predict_list[0])
                    predict_list.pop(0)
                elif index_predict > index_reference:
                    index_reference += len(reference_list[0][0])  # move
                    reference_list.pop(0)
        return correct_num

    @classmethod
    def evaluate(cls, reference, predict):
        """
        :param reference: 参考文本
        :param predict: 模型预测文本
        :return: 模型的accuracy/recall/F1_score
        """
        assert len(reference) == len(predict)
        reference_list = reference.split(" ")
        predict_list = predict.split(" ")
        accuracy = cls.__cal_accurary(reference_list, predict_list)
        recall = cls.__cal_recall(reference_list, predict_list)
        f1_score = cls.__cal_f1(accuracy, recall)
        return accuracy, recall, f1_score


if __name__ == '__main__':
    # lines = p.read_text_file("../data/trainset/test.txt")
    # # 删除空行
    # lines = p.del_blank_lines(lines)
    # for line in lines:
    #     print(line)
    # # 删除中英文标点
    # for index, line in enumerate(lines):
    #     result = p.del_num_word(p.del_special_symbol(p.del_english_word(line)))
    #     lines[index] = result
    #
    # for line in lines:
    #     print(line)
    # lines.pop()
    a = ["你好", "世界", "都", "很好", "呀", "呵呵呵"]
    b = ["你好", "世界", "都很好", "呀", "呵呵呵"]