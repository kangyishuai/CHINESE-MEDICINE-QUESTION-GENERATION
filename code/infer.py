import json
import os
import re

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer, load_vocab
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # 指定GPU

# 基本参数
n = 5               # 交叉验证
max_p_len = 194     # 篇章最大长度
max_q_len = 131     # 问题最大长度
max_a_len = 65      # 答案最大长度
head = 64           # 篇章截取中，取答案id前head个字符


# nezha配置
config_path = '../user_data/model_data/NEZHA-Large-WWM/bert_config.json'
checkpoint_path = '../user_data/model_data/NEZHA-Large-WWM/model.ckpt-346400'
dict_path = '../user_data/model_data/NEZHA-Large-WWM/vocab.txt'


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def build_model():
    """构建模型。"""
    model = build_transformer_model(
        config_path,
        checkpoint_path,
        model='nezha',
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )

    o_in = Input(shape=(None, ))
    train_model = Model(model.inputs + [o_in], model.outputs + [o_in])

    return model, train_model


class QuestionGeneration(AutoRegressiveDecoder):
    """通过beam search来生成问题。"""
    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        probas = list()
        for i in range(n):
            proba = self.models[i].predict([token_ids, segment_ids])[:, -1]
            probas.append(proba)

        return np.mean(np.concatenate(probas), axis=0, keepdims=True)

    def generate(self, passage, answer, topk=5):
        p_token_ids, _ = tokenizer.encode(passage, maxlen=max_p_len)
        a_token_ids, _ = tokenizer.encode(answer, maxlen=max_a_len)
        token_ids = p_token_ids + a_token_ids[1:]
        segment_ids = [0] * (len(p_token_ids) + len(a_token_ids[1:]))
        q_ids = self.beam_search([token_ids, segment_ids], topk)  # 基于beam search
        return tokenizer.decode(q_ids)


def do_infer(input_path, output_path):
    # 加载模型权重
    models = []
    for i in range(1, n + 1):
        model, train_model = build_model()
        train_model.load_weights(f'../user_data/model_data/fold-{i}.h5')
        models.append(model)

    # 问题生成器
    qag = QuestionGeneration(
        models, start_id=None, end_id=tokenizer._token_dict['？'],
        maxlen=max_q_len
    )

    # 加载并遍历预测数据
    data = json.load(open(input_path))
    func = lambda m: '' if len(m.group(0)) > 5 else m.group(0)
    for d in tqdm(data, desc=u'正在预测(共%s条样本)' % len(data)):
        # 篇章数据清洗
        passage = re.sub(r'\s', '', d['text'])
        passage = re.sub(r'\\n', '', passage)
        passage = re.sub(r'\([A-Za-z]+\)', func, passage)
        passage = re.sub(r'（[A-Za-z]+）', func, passage)
        for qa in d['annotations']:
            # 答案数据清洗
            answer = re.sub(r'\s', '', qa['A'])
            answer = re.sub(r'\\n', '', answer)
            answer = re.sub(r'\([A-Za-z]+\)', func, answer)
            answer = re.sub(r'（[A-Za-z]+）', func, answer)
            answer = answer[:-1] if answer.endswith('.') else answer
            # 长文本数据截断
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
            qa['Q'] = qag.generate(p, a)  # 问题生成

    # 保存结果
    json.dump(data, open(output_path, 'w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    do_infer('../data/juesai_1011.json', '../prediction_result/result.json')
