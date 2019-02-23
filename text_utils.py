#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: text_utils.py
# Author: lizhen21@baidu.com
# Date: 19-2-22

import json
from tqdm import tqdm


class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path):
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v:k for k,v in self.encoder.items()}

    def encode(self, texts, verbose=True):
        # input : trX1: [sentence1, sentence2...]
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                # sentence1
                text_tokens = []
                for token in text:
                    text_tokens.append(self.encoder.get(token, 0))
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text_tokens = []
                for token in text:
                    text_tokens.append(self.encoder.get(token, 0))
                texts_tokens.append(text_tokens)
        return texts_tokens
        # texts_tokens = [[id1_sen1, id2_sen1,...],[id1_sen2, id2_sen2],...]
