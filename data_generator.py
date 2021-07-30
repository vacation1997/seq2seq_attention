import logging as log
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical


def read_seq2seq_corups(data_file):
    """
    读取语料信息

    data_file:读取语料文件路径

    input_texts:语料输入值，古诗上句
    target_texts:语料输出值，古诗下句
    """
    input_texts = []
    target_texts = []
    for line in open(data_file, 'r', encoding='utf-8'):
        if not line.strip():
            continue
        try:
            input_text, target_text = line.split('\t')
            input_texts.append(input_text)
            target_texts.append('^' + target_text + '$')
        except ValueError:
            log.error(f'Error line: {line}')
            input_text = ''
            target_text = ''

    return input_texts, target_texts


def split_seq2seq_corpus(input_texts, target_texts, split_rate=0.2):
    """
    分隔读取到的语料

    input_texts:语料输入值，古诗上句
    target_texts:语料输出值，古诗下句
    split_rate:训练集和验证集的分割比例

    (before_input, before_target):训练集输入输出
    (after_input, after_target):验证集输入输出
    """
    total = len(input_texts)
    indices = list(range(total))
    np.random.shuffle(indices)

    input_texts = np.array(input_texts)[indices]
    target_texts = np.array(target_texts)[indices]

    split = int(total * split_rate)
    before_input, after_input = input_texts[split:], input_texts[:split]
    before_target, after_target = target_texts[split:], target_texts[:split]

    return (before_input, before_target), (after_input, after_target)


class Seq2SeqData:
    """seq2seq数据集类"""

    def __init__(self, input_texts, target_texts):
        self.encoder_tokenizer = Tokenizer(filters='', char_level=True)
        self.decoder_tokenizer = Tokenizer(filters='', char_level=True)
        self.encoder_tokenizer.fit_on_texts(input_texts)
        self.decoder_tokenizer.fit_on_texts(target_texts)

    def data_codec(self, datas, maxlen, tokenizer, one_hot=False):
        """
        数据集编码
        参数：data 数据集
             tokenizer keras中的字符转化工具
        返回值：np数组 二维或是三维（区别于encode或是decode）
        """
        fill_value = 0
        codes = []
        for data in datas:
            code = tokenizer.texts_to_sequences(data)
            code = [c[0] for c in code]
            if one_hot:
                code = to_categorical(
                    code[1:], num_classes=len(tokenizer.word_index)+1)
            codes.append(code)
        if one_hot and codes:
            fill_value = np.zeros_like(codes[0][0])
        codes = pad_sequences(codes, maxlen=maxlen,
                              padding='post', value=fill_value)

        return codes

    def generator(self, encoder_data, decoder_data, batch_size=0):
        """
        模型训练用生成器

        encoder_data:古诗上句
        decoder_data:古诗下句

        [enc_batch_data, dec_batch_data], target_batch_data:训练模型的批次输入和输出
        """
        total = len(encoder_data)
        batch_size = batch_size if batch_size > 0 else total
        emaxlen = max([len(e) for e in encoder_data])
        dmaxlen = max([len(d) for d in decoder_data])
        maxlen = max(emaxlen, dmaxlen)
        enc_batch_data, dec_batch_data = [], []
        while True:
            idxs = list(range(total))
            np.random.shuffle(idxs)
            for i, idx in enumerate(idxs):
                enc_batch_data.append(encoder_data[idx])
                dec_batch_data.append(decoder_data[idx])
                if i % batch_size == (batch_size - 1) or idx == idxs[-1]:
                    enc_batch_data = self.data_codec(
                        enc_batch_data, maxlen, self.encoder_tokenizer)
                    target_batch_data = self.data_codec(
                        dec_batch_data, maxlen, self.decoder_tokenizer, one_hot=True)
                    dec_batch_data = self.data_codec(
                        dec_batch_data, maxlen, self.decoder_tokenizer)
                    yield [enc_batch_data, dec_batch_data], target_batch_data
                    enc_batch_data, dec_batch_data = [], []
