#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vocab.py
# Author: lizhen21@baidu.com
# Date: 19-2-22

import os
import json
import codecs


def vocab_process(data_dir, size=50000):
    f = codecs.open(os.path.join(data_dir, 'baike_qa_train.json'), 'r', encoding='utf-8')
    vocab = dict()

    for line in f:
        data = json.loads(line)
        ans = '。'.join(data["answer"].strip().split())
        desc = '。'.join(data["desc"].strip().split())
        if desc != '' and ans != '':
            for word in desc:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            for word in ans:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

    f.close()
    sort_vocab = dict()
    i = 1
    for word, _ in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:size]:
        sort_vocab[word] = i
        i += 1
    with codecs.open(os.path.join(data_dir, 'vocab.json'), "w", encoding='utf-8') as f:
        json.dump(sort_vocab, f)
    return None


if __name__ == '__main__':
    vocab_process('baike_qa2019', size=50000)
