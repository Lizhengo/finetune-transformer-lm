#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: lizhen21@baidu.com
# Date: 19-2-22

import os
import time
import math
import json
import joblib
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from datasets import data_process
from text_utils import TextEncoder
from utils import encode_dataset, flatten, iter_data, find_trainable_variables, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}

lr_schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_linear':warmup_linear,
    'warmup_constant':warmup_constant,
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, resid_pdrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    h = tf.reduce_sum(e, 2)
    return h

def clf(x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
    # x.size = [n_batch_train/n_gpu * 2, n_embd]
    # ny = 1
    with tf.variable_scope('clf'):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [nx, ny], initializer=w_init)
        b = tf.get_variable("b", [ny], initializer=b_init)
        return tf.matmul(x, w)+b

def model(X, M, Y, train=False, reuse=False):
    # X = tf.placeholder(tf.int32, [n_batch_train/n_gpu, 2, n_ctx, 2])
    # M = tf.placeholder(tf.float32, [n_batch_train/n_gpu, 2, n_ctx])
    # Y = tf.placeholder(tf.int32, [n_batch_train/n_gpu])
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [n_vocab+n_special+n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        X = tf.reshape(X, [-1, n_ctx, 2])
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we)
        # h = tf.placeholder(tf.int32, [-1, n_ctx, 2, n_embd])
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        # lm_h.size = [n_batch_train/n_gpu * 2 * (n_ctx-1), n_embd]
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        # lm_logits.size = [n_batch_train/n_gpu * 2 * n_ctx, n_vocab+n_special+n_ctx]
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        # lm_losses.size = [n_batch_train/n_gpu * 2, n_ctx-1]
        lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
        # lm_losses.size = [n_batch_train/n_gpu * 2, 1]

        clf_h = tf.reshape(h, [-1, n_embd])
        # clf_h.size = [n_batch_train/n_gpu * 2 * n_ctx, n_embd]
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*n_ctx+pool_idx)

        clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        # clf_h.size = [n_batch_train/n_gpu, 2, n_embd]
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1-clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        # clf_h.size = [n_batch_train/n_gpu * 2, n_embd]
        clf_logits = clf(clf_h, 1, train=train)
        # clf_logits.size = [n_batch_train/n_gpu * 2, 1]
        clf_logits = tf.reshape(clf_logits, [-1, 2])
        # clf_logits.size = [n_batch_train/n_gpu, 2]
        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        # clf_losses.size = [n_batch_train/n_gpu]
        return clf_logits, clf_losses, lm_losses

def mgpu_train(*xs):
    # xs[0]: X_train = tf.placeholder(tf.int32, [n_batch_train, 2, n_ctx, 2])
    # xs[1]: M_train = tf.placeholder(tf.float32, [n_batch_train, 2, n_ctx])
    # xs[2]: Y_train = tf.placeholder(tf.int32, [n_batch_train])
    gpu_ops = []
    gpu_grads = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)

    for i, xs in enumerate(zip(*xs)):
        # i = 3, xs = [xs[0]_i, x[1]_i, x[2]_i]
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
            clf_logits, clf_losses, lm_losses = model(*xs, train=True, reuse=do_reuse)
            # clf_losses.size = [n_batch_train/n_gpu]
            # clf_logits.size = [n_batch_train/n_gpu, 2]
            # lm_losses.size = [n_batch_train/n_gpu * 2, 1]

            if clf_coef > 0:
                train_loss = tf.reduce_mean(lm_losses) + clf_coef*tf.reduce_mean(clf_losses)
            else:
                train_loss = tf.reduce_mean(lm_losses)
            params = find_trainable_variables("model")
            grads = tf.gradients(train_loss, params)
            grads = list(zip(grads, params))
            gpu_grads.append(grads)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    grads = average_grads(gpu_grads)
    grads = [g for g, p in grads]
    train = opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)
    return [train]+ops

def mgpu_predict(*xs):
    gpu_ops = []
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
            clf_logits, clf_losses, lm_losses = model(*xs, train=False, reuse=True)
            gpu_ops.append([clf_logits, clf_losses, lm_losses])
    ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
    return ops

def transform_roc(X1, X2, X3):
    # X1: [[id1_sen1, id2_sen1, ...], [id1_sen2, id2_sen2], ...]
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        x13 = [start]+x1[:max_len]+[delimiter]+x3[:max_len]+[clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb

def iter_apply(Xs, Ms, Ys):
    fns = [lambda x:np.concatenate(x, 0), lambda x:float(np.sum(x)), lambda x:float(np.sum(x))]
    results = []
    for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            res = sess.run([eval_mgpu_logits, eval_mgpu_clf_loss, eval_mgpu_lm_loss],
                           {X_train:xmb, M_train:mmb, Y_train:ymb})
        else:
            res = sess.run([eval_logits, eval_clf_loss, eval_lm_loss],
                           {X:xmb, M:mmb, Y:ymb})
        res = [r*n for r in res]
        results.append(res)
    results = zip(*results)
    # results = [(eval_mgpu_logits1*n, eval_mgpu_logits2*n...), (eval_mgpu_clf_loss1*n, eval_mgpu_clf_loss2*n...),
    #            (eval_mgpu_lm_loss1*n, eval_mgpu_lm_loss2*n...)]
    return [fn(res) for res, fn in zip(results, fns)]

def iter_predict(Xs, Ms):
    logits = []
    for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
        n = len(xmb)
        if n == n_batch_train:
            logits.append(sess.run(eval_mgpu_logits, {X_train:xmb, M_train:mmb}))
        else:
            logits.append(sess.run(eval_logits, {X:xmb, M:mmb}))
    logits = np.concatenate(logits, 0)
    return logits

def save(path):
    ps = sess.run(params)
    joblib.dump(ps, make_path(path))

def log():
    global best_ppl
    tr_clf_logits, tr_clf_cost, tr_lm_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_clf_logits, va_clf_cost, va_lm_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_lm_cost/len(trY[:n_valid])
    va_cost = va_lm_cost/n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_clf_logits, 1))*100.
    va_acc = accuracy_score(vaY, np.argmax(va_clf_logits, 1))*100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=np.exp(tr_cost),
               va_cost=np.exp(va_cost), tr_acc=tr_acc, va_acc=va_acc)
    print('\nn_epochs: %d , n_updates: %d , tr_ppl: %.3f , va_ppl: %.3f , tr_clf_acc: %.2f , val_clf_acc: %.2f\n'
          %(n_epochs, n_updates, np.exp(tr_cost), np.exp(va_cost), tr_acc, va_acc))

    ppl = np.exp(va_cost)

    if ppl < best_ppl or best_ppl < 0:
        best_ppl = ppl
        best_sv_name = os.path.join(save_dir, desc + '-best')
        print("Saving best model to %s." % save_dir)
        sv.save(sess, best_sv_name)
        # save(os.path.join(save_dir, desc, 'best_params.jl'))

argmax = lambda x:np.argmax(x, 1)

pred_fns = {
    'rocstories':argmax,
}

filenames = {
    'rocstories':'ROCStories.tsv',
}

label_decoders = {
    'rocstories':None,
}

def predict():
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='gpt-baike-qa')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='baike_qa2019/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=64)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='baike_qa2019/vocab.json')
    parser.add_argument('--clf_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)

    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(encoder_path)
    encoder = text_encoder.encoder
    # encoder = json.load(open(encoder_path)), ex. {".": 1, ",": 2, "t": 3, "h": 4, ...}
    n_vocab = len(text_encoder.encoder)
    print('vocab size: ', n_vocab)

    (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = encode_dataset(data_process(data_dir),
                                                                                          encoder=text_encoder)
    # trX1/trX2/trX3: [[id1_sen1, id2_sen1,...],[id1_sen2, id2_sen2],...]
    # trY: [0, 1, 1,....]
    n_y = 2
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3
    max_len = n_ctx//2-2
    n_ctx = min(max([len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]+[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]+[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)])+3, n_ctx)
    trX, trM = transform_roc(trX1, trX2, trX3)
    # trX: [n_batch, 2, n_ctx, 2], dtype=np.int32
    # trM: [n_batch, 2, n_ctx], dtype=np.float32
    vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
    teX, teM = transform_roc(teX1, teX2, teX3)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = n_batch*n_gpu
    n_updates_total = (n_train//n_batch_train)*n_iter

    X_train = tf.placeholder(tf.int32, [n_batch_train, 2, n_ctx, 2])
    M_train = tf.placeholder(tf.float32, [n_batch_train, 2, n_ctx])
    X = tf.placeholder(tf.int32, [None, 2, n_ctx, 2])
    M = tf.placeholder(tf.float32, [None, 2, n_ctx])

    Y_train = tf.placeholder(tf.int32, [n_batch_train])
    Y = tf.placeholder(tf.int32, [None])

    train, logits, clf_losses, lm_losses = mgpu_train(X_train, M_train, Y_train)
    # clf_losses.size = [n_batch_train]
    # clf_logits.size = [n_batch_train, 2]
    # lm_losses.size = [n_batch_train * 2, 1]
    clf_loss = tf.reduce_mean(clf_losses)
    lm_loss = tf.reduce_mean(lm_losses)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    restore_v = dict()
    for v in tf.trainable_variables():
        print("store:", v.name)
        restore_v[v.name] = v
    params = find_trainable_variables("model")
    sv = tf.train.Saver(restore_v)
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading param from %s" % ckpt.model_checkpoint_path)
        sv.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh param.")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    eval_mgpu_logits, eval_mgpu_clf_losses, eval_mgpu_lm_losses = mgpu_predict(X_train, M_train, Y_train)
    eval_mgpu_clf_loss = tf.reduce_mean(eval_mgpu_clf_losses)
    eval_mgpu_lm_loss = tf.reduce_mean(eval_mgpu_lm_losses)

    eval_logits, eval_clf_losses, eval_lm_losses = model(X, M, Y, train=False, reuse=True)
    eval_clf_loss = tf.reduce_mean(eval_clf_losses)
    eval_lm_loss = tf.reduce_mean(eval_lm_losses)

    n_updates = 0
    n_epochs = 0

    best_ppl = -1
    for i in range(n_iter):
        for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trY, random_state=np.random), n_batch=n_batch_train, truncate=True, verbose=True):
            cost, _ = sess.run([lm_loss, train], {X_train:xmb, M_train:mmb, Y_train:ymb})
            n_updates += 1
            if n_updates % (max((n_train//n_batch_train) // 10, 1)) == 0:
                log()
                sv_name = os.path.join(save_dir, desc)
                print("Saving model to %s." % save_dir)
                sv.save(sess, sv_name, global_step=n_updates)
        n_epochs += 1
        log()
