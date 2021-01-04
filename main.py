import json
import re

import keras.backend.tensorflow_backend as KTF
import numpy as np
import pandas as pd
import tensorflow as tf
from bert4keras.backend import keras, K, search_layer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer, load_vocab
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from rouge import Rouge  # pip install rouge
from sklearn.model_selection import KFold
from tqdm import tqdm

# 基本参数
n = 5               # 交叉验证
max_p_len = 194     # 最大篇章长度
max_q_len = 131     # 最大问题长度
max_a_len = 65      # 最大答案长度
head = 64           # 取答案开头位置之前的文本长度
batch_size = 4
epochs = 20
SEED = 2020

# nezha配置
config_path = 'NEZHA-Large-WWM/bert_config.json'
checkpoint_path = 'NEZHA-Large-WWM/model.ckpt-346400'
dict_path = 'NEZHA-Large-WWM/vocab.txt'

# 其他配置
seps = u'\n。！？!?；;，, '


def json2df(dic):
    """Json转Pandas的DataFrame。"""

    D = {'passage': list(), 'question': list(), 'answer': list()}
    for d in dic:
        for qa in d['annotations']:
            D['passage'].append(d['text'])
            D['question'].append(qa['Q'])
            D['answer'].append(qa['A'])

    return pd.DataFrame(D)


def load_data(filename):
    """加载数据。"""

    train = json.load(open(filename))
    df_train = json2df(train)

    # 剔除空白字符
    df_train = df_train.applymap(lambda x: re.sub(r'\s', '', x))
    df_train = df_train.applymap(lambda x: re.sub(r'\\n', '', x))

    # 剔除带括号的英文
    func = lambda m: '' if len(m.group(0)) > 5 else m.group(0)
    df_train = df_train.applymap(lambda x: re.sub(r'\([A-Za-z]+\)', func, x))
    df_train = df_train.applymap(lambda x: re.sub(r'（[A-Za-z]+）', func, x))

    # 筛选出答案与篇章不匹配的数据
    tmp = []
    for idx, row in df_train.iterrows():
        if row['answer'] not in row['passage']:
            tmp.append(idx)

    # 处理部分不匹配数据
    no_match = df_train.loc[tmp]
    df_train.drop(index=tmp, inplace=True)
    no_match['answer'] = no_match['answer'].map(lambda x: x.replace('.', ''))
    df_train = pd.concat([df_train, no_match])
    df_train.reset_index(drop=True, inplace=True)

    # 文本截断
    D = list()
    for _, row in df_train.iterrows():
        if len(row['passage']) < max_p_len - 2 and len(
                row['answer']) < max_a_len - 1:
            D.append((row['passage'], row['question'], row['answer']))
        else:
            a = row['answer'][:max_a_len - 1] if len(
                row['answer']) > max_a_len - 1 else row['answer']
            try:
                idx = row['passage'].index(a)
                if len(row['passage'][idx:]) < (max_p_len - 2 - head):
                    p = row['passage'][-(max_p_len - 2):]
                else:
                    p = row['passage'][max(0, idx - head):]
                    p = p[:max_p_len - 2]
            except ValueError:
                p = row['passage'][:max_p_len - 2]
            D.append((p, row['question'], a))

    return D


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器。"""

    def __init__(self, data, batch_size=32, buffer_size=None, random=False):
        super().__init__(data, batch_size, buffer_size)
        self.random = random

    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
        """

        batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []
        for is_end, (p, q, a) in self.sample(random):
            p_token_ids, _ = tokenizer.encode(p, maxlen=max_p_len)
            a_token_ids, _ = tokenizer.encode(a, maxlen=max_a_len)
            q_token_ids, _ = tokenizer.encode(q, maxlen=max_q_len)
            token_ids = p_token_ids + a_token_ids[1:] + q_token_ids[1:]
            segment_ids = [0] * (len(p_token_ids) + len(a_token_ids[1:]))
            segment_ids += [1] * (len(token_ids) - len(p_token_ids) - len(
                a_token_ids[1:]))

            # 随机替换
            o_token_ids = token_ids
            if np.random.random() > 0.5:
                token_ids = [
                    t if s == 0 or (s == 1 and np.random.random() > 0.3)
                    else np.random.choice(token_ids)
                    for t, s in zip(token_ids, segment_ids)
                ]

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_o_token_ids.append(o_token_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_o_token_ids = sequence_padding(batch_o_token_ids)
                yield [batch_token_ids, batch_segment_ids, batch_o_token_ids], None
                batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d


def build_model():
    """构建模型。"""

    # 预测用模型
    model = build_transformer_model(
        config_path,
        checkpoint_path,
        model='nezha',
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )

    # 训练用模型
    o_in = Input(shape=(None,))
    train_model = Model(model.inputs + [o_in], model.outputs + [o_in])

    # 交叉熵作为loss，并mask掉输入部分的预测
    y_true = train_model.input[2][:, 1:]    # 目标tokens
    y_mask = train_model.input[1][:, 1:]
    y_pred = train_model.output[0][:, :-1]  # 预测tokens，预测与目标错开一位
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    train_model.add_loss(cross_entropy)
    train_model.compile(optimizer=Adam(1e-5))

    return model, train_model


def adversarial_training(model, embedding_name, epsilon=1.):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


class QuestionGeneration(AutoRegressiveDecoder):
    """通过beam search来生成问题（训练用）。"""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, passage, answer, topk=5):
        p_token_ids, _ = tokenizer.encode(passage, maxlen=max_p_len)
        a_token_ids, _ = tokenizer.encode(answer, maxlen=max_a_len)
        token_ids = p_token_ids + a_token_ids[1:]
        segment_ids = [0] * (len(p_token_ids) + len(a_token_ids[1:]))
        q_ids = self.beam_search([token_ids, segment_ids], topk)  # 基于beam search
        return tokenizer.decode(q_ids)


class QuestionGenerationV2(AutoRegressiveDecoder):
    """通过beam search来生成问题（预测用）"""

    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        probas = []
        for i in range(n):
            proba = self.models[i].predict([token_ids, segment_ids])[:, -1]
            probas.append(proba)

        pred = np.zeros_like(probas[0])
        for proba in probas:
            pred += proba / len(proba)

        return pred

    def generate(self, passage, answer, topk=5):
        p_token_ids, _ = tokenizer.encode(passage, maxlen=max_p_len)
        a_token_ids, _ = tokenizer.encode(answer, maxlen=max_a_len)
        token_ids = p_token_ids + a_token_ids[1:]
        segment_ids = [0] * (len(p_token_ids) + len(a_token_ids[1:]))
        q_ids = self.beam_search([token_ids, segment_ids],
                                 topk)  # 基于beam search
        return tokenizer.decode(q_ids)


class Evaluator(keras.callbacks.Callback):
    def __init__(self, valid_data, qg):
        self.rouge = Rouge()
        self.best_rouge_l = 0.
        self.valid_data = valid_data
        self.qg = qg
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        rouge_l = self.evaluate(self.valid_data)  # 评测模型
        if rouge_l > self.best_rouge_l:
            self.best_rouge_l = rouge_l
        logs['val_rouge_l'] = rouge_l
        print(f'val_rouge_l: {rouge_l:.5f}, '
              f'best_val_rouge_l: {self.best_rouge_l:.5f}')

    def evaluate(self, data, topk=1):
        total = 0
        rouge_l = 0
        for p, q, a in tqdm(data):
            total += 1
            q = ' '.join(q)
            pred_q = ' '.join(self.qg.generate(p, a, topk))
            if pred_q.strip():
                scores = self.rouge.get_scores(hyps=pred_q, refs=q)
                rouge_l += scores[0]['rouge-l']['f']

        rouge_l /= total

        return rouge_l


def do_train(filename):
    """模型训练。"""

    data = load_data(filename)

    kf = KFold(n_splits=n, shuffle=True, random_state=SEED)     # 交叉验证
    for fold, (trn_idx, val_idx) in enumerate(kf.split(data), 1):
        print(f'Fold {fold}')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
        sess = tf.Session(graph=tf.get_default_graph(), config=config)
        KTF.set_session(sess)

        # 划分训练、验证集
        train_data = [data[i] for i in trn_idx]
        valid_data = [data[i] for i in val_idx]

        # 训练数据生成器
        train_generator = data_generator(train_data, batch_size, random=True)

        # 构建模型
        model, train_model = build_model()

        # 对抗训练
        adversarial_training(train_model, 'Embedding-Token', 0.5)

        # 解码器
        qg = QuestionGeneration(
            model, start_id=None, end_id=tokenizer._token_dict['？'],
            maxlen=max_q_len
        )

        callbacks = [
            Evaluator(valid_data, qg),
            EarlyStopping(
                monitor='val_rouge_l',
                patience=1,
                verbose=1,
                mode='max'),
            ModelCheckpoint(
                f'fold-{fold}.h5',
                monitor='val_rouge_l',
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
                mode='max'),
        ]

        train_model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=callbacks,
        )

        KTF.clear_session()
        sess.close()


def do_predict(filename, output_filename='submission.json'):
    """结果预测。"""

    # 加载模型
    models = []
    for i in range(1, n + 1):
        model, train_model = build_model()
        train_model.load_weights(f'fold-{i}.h5')
        models.append(model)

    # 解码器
    qag = QuestionGenerationV2(
        models, start_id=None, end_id=tokenizer._token_dict['？'],
        maxlen=max_q_len
    )

    data = json.load(open(filename))
    func = lambda m: '' if len(m.group(0)) > 5 else m.group(0)
    for d in tqdm(data, desc=u'正在预测(共%s条样本)' % len(data)):
        # 篇章预处理
        passage = re.sub(r'\s', '', d['text'])
        passage = re.sub(r'\\n', '', passage)
        passage = re.sub(r'\([A-Za-z]+\)', func, passage)
        passage = re.sub(r'（[A-Za-z]+）', func, passage)
        for qa in d['annotations']:
            # 答案预处理
            answer = re.sub(r'\s', '', qa['A'])
            answer = re.sub(r'\\n', '', answer)
            answer = re.sub(r'\([A-Za-z]+\)', func, answer)
            answer = re.sub(r'（[A-Za-z]+）', func, answer)
            answer = answer[:-1] if answer.endswith('.') else answer
            # 文本截断
            if len(passage) < max_p_len - 2 and len(answer) < max_a_len - 1:
                a = answer
                p = passage
            else:
                a = answer[:max_a_len - 1] if len(
                    answer) > max_a_len - 1 else answer
                try:
                    idx = passage.index(a)
                    if len(passage[idx:]) < (max_p_len - 2 - head):
                        p = passage[-(max_p_len - 2):]
                    else:
                        p = passage[max(0, idx - head):]
                        p = p[:max_p_len - 2]
                except ValueError:
                    p = passage[:max_p_len - 2]
            # 生成问题
            qa['Q'] = qag.generate(p, a)

    # 保存结果
    json.dump(data, open(output_filename, 'w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':

    do_train('data/round1_train_0907.json')     # 训练

    do_predict('data/round1_test_0907.json')    # 预测
