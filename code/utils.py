import json
import re
from collections import defaultdict

import pandas as pd


def json2df(filename):
    """json转pandas.DataFrame。"""
    json_data = json.load(open(filename))
    D = defaultdict(list)
    for d in json_data:
        for qa in d['annotations']:
            D['passage'].append(d['text'])
            D['question'].append(qa['Q'])
            D['answer'].append(qa['A'])

    return pd.DataFrame(D)


def preprocess(df):
    """数据预处理。"""
    # 剔除空白字符
    df = df.applymap(lambda x: re.sub(r'\s', '', x))
    df = df.applymap(lambda x: re.sub(r'\\n', '', x))

    # 剔除带括号的英文
    func = lambda m: '' if len(m.group(0)) > 5 else m.group(0)
    df = df.applymap(lambda x: re.sub(r'\([A-Za-z]+\)', func, x))
    df = df.applymap(lambda x: re.sub(r'（[A-Za-z]+）', func, x))

    # 筛选出答案与篇章不匹配的数据
    tmp = list()
    for idx, row in df.iterrows():
        if row['answer'] not in row['passage']:
            tmp.append(idx)

    # 处理部分不匹配数据
    no_match = df.loc[tmp]
    df.drop(index=tmp, inplace=True)
    no_match['answer'] = no_match['answer'].map(lambda x: x.replace('.', ''))
    df = pd.concat([df, no_match])
    df.reset_index(drop=True, inplace=True)

    return df
