#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: datasets.py
# Author: lizhen21@baidu.com
# Date: 19-2-22

import os
import csv
import numpy as np
import json
from tqdm import tqdm


seed = 3535999445


def _data_process(path):
    f = open(path, 'r', encoding='utf-8')

    st = []
    ct1 = []
    y = []

    for line in f:
        data = json.loads(line)
        ans = '。'.join(data["answer"].strip().split())
        desc = '。'.join(data["desc"].strip().split())
        if desc != '' and ans != '':
            st.append(desc)
            ct1.append(ans)
            y.append(0)
    ct2 = ct1[1:] + [ct1[0]]
    return st, ct1, ct2, y


def data_process(data_dir):
    tr_q, tr_a1, tr_a2, tr_ys = _data_process(os.path.join(data_dir, 'baike_qa_train.json'))
    va_q, va_a1, va_a2, va_ys = _data_process(os.path.join(data_dir, 'baike_qa_test.json'))
    teX1, teX2, teX3, _ = _data_process(os.path.join(data_dir, 'baike_qa_test.json'))

    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_q, tr_a1, tr_a2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_q, va_a1, va_a2, va_ys ):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)
    # (trX1, trX2, trX3, trY) = ([sentence1, sentence2...],[quiz1_1, quiz1_2....],[quiz2_1, quiz2_2...],[0,1,1,0.....])
