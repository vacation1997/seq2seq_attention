import math
import keras as K
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate
from keras.layers import Attention

from data_generator import read_seq2seq_corups, split_seq2seq_corpus, Seq2SeqData

import logging as log


def build_seq2seq_attention_train_model(encoder_input_dim, decoder_input_dim, embedding_dimention):
    """
    构建训练模型

    encoder_input_dim:输入字典长度
    decoder_input_dim:输出字典长度
    embedding_dimention:embedding层的维度值

    model:seq2seq_attention训练模型
    """
    # encoder
    encoder_input = Input(shape=(None,))
    encoder_embedding = Embedding(
        encoder_input_dim, embedding_dimention, name='encoder_embedding')(encoder_input)
    encoder_bilstm = Bidirectional(
        LSTM(
            lstm_hidden,
            return_state=True,
            dropout=0.05
        ), name='encoder_bilstm'
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(
        encoder_embedding)
    encoder_states = Concatenate()(
        [forward_h, backward_h]), Concatenate()([forward_c, backward_c])

    # decoder
    decoder_input = Input(shape=(None,))
    decoder_embedding = Embedding(
        decoder_input_dim, embedding_dimention, name='decoder_embedding')(decoder_input)
    decoder_lstm = LSTM(lstm_hidden*2, return_state=True,
                        return_sequences=True, dropout=0.05, name='decoder_lstm')
    decoder_outputs, *_ = decoder_lstm(
        decoder_embedding, initial_state=encoder_states)

    # attention
    attention_layer = Attention(name='attention')
    query_value_attention_seq = attention_layer(
        [decoder_outputs, encoder_outputs])
    decoder_concat_input = Concatenate()(
        [decoder_outputs, query_value_attention_seq])
    decoder_dense = Dense(
        decoder_input_dim, activation='softmax', name='decoder_dense')
    final_output = decoder_dense(decoder_concat_input)

    # 构建模型
    model = Model([encoder_input, decoder_input], final_output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['acc'])

    return model


def fit_seq2seq_attention_train_model(model, train_datas, valid_datas, seq2seqdata, batch_size, epochs, callbacks=None):
    """
    训练seq2seq_attention训练模型

    model:传入seq2seq_attention训练模型
    train_datas:训练集输入输出
    valid_datas:验证集输入输出
    seq2seqdata:生成器类对象
    batch_size:训练的批次大小
    epochs:训练的轮数
    callbacks:传入回调列表

    hist:训练后的seq2seq_attention模型
    """
    encoder_datas, decoder_datas = train_datas
    train_gen = seq2seqdata.generator(encoder_datas, decoder_datas, batch_size)
    encoder_valids, decoder_valids = valid_datas
    valid_gen = seq2seqdata.generator(
        encoder_valids, decoder_valids, batch_size)

    train_steps = int(math.ceil(len(encoder_datas) / batch_size))
    valid_steps = int(math.ceil(len(encoder_valids) / batch_size))

    hist = model.fit(train_gen, steps_per_epoch=train_steps, epochs=epochs,
                     validation_data=valid_gen, validation_steps=valid_steps, callbacks=callbacks)

    return hist


def build_seq2seq_attention_inference_model(model_path):
    """
    创建seq2seq_attention预测模型（callbacks可能含有保存模型的效果）

    model_path:文件保存路径

    seq2seq_attention_model:seq2seq_attention预测模型（散装）
    """
    model = load_model(model_path)

    # encoder
    encoder_inputs = K.layers.Input(shape=(None,))
    encoder_embedding = model.get_layer('encoder_embedding')(encoder_inputs)
    encoder_bilstm = model.get_layer('encoder_bilstm')
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(
        encoder_embedding)
    encoder_states = [Concatenate()([forward_h, backward_h]),
                      Concatenate()([forward_c, backward_c])]
    encoder_model = Model(
        encoder_inputs, [encoder_outputs] + encoder_states)

    # decoder
    decoder_inputs = K.layers.Input(shape=(None,))
    decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_state_h = K.layers.Input(shape=(lstm_hidden * 2,))
    decoder_state_c = K.layers.Input(shape=(lstm_hidden * 2,))
    decoder_states_inputs = [decoder_state_h, decoder_state_c]
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, * \
        decoder_states = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # attention
    dec_out_input = Input(shape=(None, lstm_hidden * 2,))
    enc_out_input = Input(shape=(None, lstm_hidden * 2,))
    attention_layer = model.get_layer('attention')
    attention_output = attention_layer([dec_out_input, enc_out_input])
    decoder_concat_input = Concatenate()([dec_out_input, attention_output])
    dec_dense = model.get_layer('decoder_dense')
    final_output = dec_dense(decoder_concat_input)
    inf_model = Model([dec_out_input, enc_out_input], final_output)

    # 构建模型
    seq2seq_attention_model = {'encoder_model': encoder_model,
                               'decoder_model': decoder_model, 'inf_model': inf_model}
    return seq2seq_attention_model


def predict_(encoder_input, seq2seq_attention_model, input_token, target_token, max_target_length):
    """
    预测

    encoder_input:需要预测的序列
    seq2seq_attention_model:seq2seq_attention预测模型
    input_token:生成器对象的模型输入的tokenizer
    target_token:生成器对象的模型输出的tokenizer
    max_target_length:最大目标生成长度

    decoded_translation:预测出的序列
    """
    txt = [input_token.word_index[i] for i in encoder_input]
    enc_op, *stat = seq2seq_attention_model['encoder_model'].predict([txt])
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = target_token.word_index['^']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = seq2seq_attention_model['decoder_model'].predict(
            [empty_target_seq] + stat)
        final_output = seq2seq_attention_model['inf_model'].predict(
            [dec_outputs, enc_op])
        sampled_word_index = np.argmax(final_output[0, 0, :])
        sampled_word = target_token.index_word[sampled_word_index]
        if sampled_word != '$':
            decoded_translation += sampled_word
        if sampled_word == '$' or len(decoded_translation) > max_target_length:
            stop_condition = True
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        stat = [h, c]

    return decoded_translation


if __name__ == '__main__':
    log.basicConfig(level=log.DEBUG)
    lstm_hidden, embedding_dimention, batch_size, epochs = 200, 400, 32, 10
    model_file = 'bast_attention_model.h5'
    poetry_file = 'poetry_corpus.txt'
    input_text, target_text = read_seq2seq_corups(poetry_file)
    seq2seq_data = Seq2SeqData(input_text, target_text)

    (train_input, train_target), (valid_input,
                                  valid_target) = split_seq2seq_corpus(input_text, target_text)
    encoder_input_dim = len(seq2seq_data.encoder_tokenizer.word_index) + 1
    decoder_input_dim = len(seq2seq_data.decoder_tokenizer.word_index) + 1
    callback_list = [
        K.callbacks.ModelCheckpoint(model_file, save_best_only=True),
        K.callbacks.EarlyStopping(monitor='loss', patience=5)
    ]
    model = build_seq2seq_attention_train_model(
        encoder_input_dim, decoder_input_dim, embedding_dimention)

    fit_seq2seq_attention_train_model(
        model,
        (train_input, train_target),
        (valid_input, valid_target),
        seq2seq_data,
        batch_size,
        epochs,
        callback_list)

    # 模型推理
    # seq2seq_attention_model = build_seq2seq_attention_inference_model(
    #     model_file)
    # encoder_input = input('输入一段文字：')
    # result = predict_(
    #     encoder_input,
    #     seq2seq_attention_model,
    #     seq2seq_data.encoder_tokenizer,
    #     seq2seq_data.decoder_tokenizer,
    #     12)
    # print(result)
