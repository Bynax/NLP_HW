# -*- coding: utf-8 -*-
# Created by bohuanshi on 2019/5/19

import re
import string
from zhon.hanzi import punctuation


class Preprocess(object):
    """数据集预处理"""

    # 用来处理数据的正则表达式
    DIGIT_RE = re.compile(r'\d+')
    LETTER_RE = re.compile(r'[a-zA-Z]+')
    SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')  # 用以删除一些特殊符号

    @staticmethod
    def read_text_file(text_file, is_read_lines=False):
        """
        读取文本文件,并返回由每行文本作为元素组成的list
        :param text_file: 文本位置
        :param is_read_lines: 是否按行读取，True->list中每个元素为一行，False->list中每个元素为一个word
        :return:
        """
        with open(text_file, 'r', encoding='utf-8') as file:
            if is_read_lines:
                lines = [line.strip() for line in file]
            else:
                lines = file.read().replace("  ", " ")
        return lines

    @staticmethod
    def __del_blank_lines(sentences):
        """删除句子列表中的空行，返回没有空行的句子列表
        Args:
            sentences: 字符串列表
        """
        return [s for s in sentences if s.split()]

    @staticmethod
    def __del_punctuation(sentence):
        """删除字符串中的中英文标点.
        Args:
            sentence: 字符串
        """
        en_punc_tab = str.maketrans('', '', string.punctuation)  # ↓ ① ℃处理不了
        sent_no_en_punc = sentence.translate(en_punc_tab)
        return re.sub(r'[%s]+' % punctuation, "", sent_no_en_punc)

    @classmethod
    def __del_special_symbol(cls, sentence):
        """删除句子中的乱码和一些特殊符号。"""
        return cls.SPECIAL_SYMBOL_RE.sub('', sentence)

    @classmethod
    def __del_english_word(cls, sentence):
        """删除句子中的英文字符"""
        return cls.LETTER_RE.sub('', sentence)

    @classmethod
    def __del_num_word(cls, sentence):
        """删除句子中的数字"""
        return cls.DIGIT_RE.sub('', sentence)

    @classmethod
    def process_text(cls, lines):
        """
        对文本进行预处理，包括删除空行、删除中英文标点、删除特殊字符、删除数字等
        :param lines:按照行读取后的行list
        :return: 处理后的文本
        """
        # 删除空行
        lines = cls.__del_blank_lines(lines)
        # 删除中英文标点
        for index, line in enumerate(lines):
            result = cls.__del_num_word(cls.__del_special_symbol(cls.__del_english_word(line)))
            lines[index] = result
        return lines
