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
    def read_text_file(text_file):
        """读取文本文件,并返回由每行文本作为元素组成的list."""
        with open(text_file, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
        return lines

    @staticmethod
    def write_text_file(text_list, target_file):
        """将文本列表写入目标文件
        Args:
            text_list: 列表，每个元素是一条文本
            target_file: 字符串，写入目标文件路径
        """
        with open(target_file, 'w', encoding='utf-8') as writer:
            for text in text_list:
                writer.write(text + '\n')

    @staticmethod
    def del_blank_lines(sentences):
        """删除句子列表中的空行，返回没有空行的句子列表

        Args:
            sentences: 字符串列表
        """
        return [s for s in sentences if s.split()]

    @staticmethod
    def del_punctuation(sentence):
        """删除字符串中的中英文标点.
        Args:
            sentence: 字符串
        """
        en_punc_tab = str.maketrans('', '', string.punctuation)  # ↓ ① ℃处理不了
        sent_no_en_punc = sentence.translate(en_punc_tab)
        return re.sub(r'[%s]+' % punctuation, "", sent_no_en_punc)

    @classmethod
    def del_special_symbol(cls, sentence):
        """删除句子中的乱码和一些特殊符号。"""
        return cls.SPECIAL_SYMBOL_RE.sub('', sentence)

    @classmethod
    def del_english_word(cls, sentence):
        """删除句子中的英文字符"""
        return cls.LETTER_RE.sub('', sentence)

    @classmethod
    def del_num_word(cls, sentence):
        """删除句子中的数字"""
        return cls.DIGIT_RE.sub('', sentence)
