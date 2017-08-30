import numpy as np
import theano
import theano.tensor as T
import lasagne

import sys
import time
import utils
import config
import logging
import nn_layers


def gen_examples(x1, x2, l, y, batch_size):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_l = l[minibatch]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y))
    return all_ex


def build_fn(args, embeddings):
    """
        Build training and testing functions.
    """
    in_x1 = T.imatrix('x1')
    in_x2 = T.imatrix('x2')
    in_mask1 = T.matrix('mask1')
    in_mask2 = T.matrix('mask2')
    in_l = T.matrix('l')
    in_y = T.ivector('y')

    l_in1 = lasagne.layers.InputLayer((None, None), in_x1)
    l_mask1 = lasagne.layers.InputLayer((None, None), in_mask1)
    l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, args.vocab_size,
                                           args.embedding_size, W=embeddings)

    l_in2 = lasagne.layers.InputLayer((None, None), in_x2)
    l_mask2 = lasagne.layers.InputLayer((None, None), in_mask2)
    l_emb2 = lasagne.layers.EmbeddingLayer(l_in2, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    network1 = nn_layers.stack_rnn(l_emb1, l_mask1, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=(args.att_func == 'last'),
                                   bidir=args.bidir,
                                   name='d',
                                   rnn_layer=args.rnn_layer)

    network2 = nn_layers.stack_rnn(l_emb2, l_mask2, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='q',
                                   rnn_layer=args.rnn_layer)

    args.rnn_output_size = args.hidden_size * 2 if args.bidir else args.hidden_size

    if args.att_func == 'mlp':
        att = nn_layers.MLPAttentionLayer([network1, network2], args.rnn_output_size,
                                          mask_input=l_mask1)
    elif args.att_func == 'bilinear':
        att_inspect = nn_layers.InspectBilinearAttentionLayer([network1, network2], args.rnn_output_size,
                                               mask_input=in_mask1)
        alphas = lasagne.layers.get_output(att_inspect, deterministic=True)
        att = nn_layers.BilinearAttentionLayer([network1, network2], alphas)
    elif args.att_func == 'avg':
        att = nn_layers.AveragePoolingLayer(network1, mask_input=l_mask1)
    elif args.att_func == 'last':
        att = network1
    elif args.att_func == 'dot':
        att = nn_layers.DotProductAttentionLayer([network1, network2], mask_input=l_mask1)
    else:
        raise NotImplementedError('att_func = %s' % args.att_func)

    network = lasagne.layers.DenseLayer(att, args.num_labels,
                                        nonlinearity=lasagne.nonlinearities.softmax)

    if args.pre_trained is not None:
        dic = utils.load_params(args.pre_trained)
        lasagne.layers.set_all_param_values(network, dic['params'], trainable=True)
        del dic['params']
        logging.info('Loaded pre-trained model: %s' % args.pre_trained)
        for dic_param in dic.iteritems():
            logging.info(dic_param)

    logging.info('#params: %d' % lasagne.layers.count_params(network, trainable=True))
    for layer in lasagne.layers.get_all_layers(network):
        logging.info(layer)

    # Test functions
    test_prob = lasagne.layers.get_output(network, deterministic=True) * in_l
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))

    output_lst = [acc, test_prediction]
    if args.att_output:
        output_lst.append(alphas)
    test_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_l, in_y], output_lst)

    # Train functions
    train_prediction = lasagne.layers.get_output(network) * in_l
    train_prediction = train_prediction / \
        train_prediction.sum(axis=1).reshape((train_prediction.shape[0], 1))
    train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, in_y).mean()
    # TODO: lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    params = lasagne.layers.get_all_params(network, trainable=True)

    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
    train_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_l, in_y],
                               loss, updates=updates)

    return train_fn, test_fn, params


def eval_acc(test_fn, all_examples, att_output):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = 0
    all_preds = []
    all_att_ws = []
    all_qs = []
    n_examples = 0
    for x1, mask1, x2, mask2, l, y in all_examples:
        if att_output:
            _acc, preds, att_ws = test_fn(x1, mask1, x2, mask2, l, y)
            # filter out the "empty" elements in att_ws, present because of batching
            for i in range(mask1.shape[0]):
                positions_true = np.where(mask1[i])
                att_ws_true = att_ws[i][positions_true]
                x1_true = x1[i][positions_true]
                assert len(att_ws_true) == len(x1_true)
                all_att_ws.append(list(zip(x1_true, att_ws_true)))  # [(id, att), ...]
            for i in range(mask2.shape[0]):
                positions_true = np.where(mask2[i])
                x2_true = x2[i][positions_true]
                all_qs.append(x2_true)
        else:
            _acc, preds = test_fn(x1, mask1, x2, mask2, l, y)
        assert len(y) == len(preds)
        acc += _acc
        all_preds.extend(preds)
        n_examples += len(x1)

    return acc * 100.0 / n_examples, all_preds, all_att_ws, all_qs


def to_output_preds(ids, preds, inv_entity_dict, relabeling):
    """
    :param ids: [(id: {@entity0: @entityAnswer})] if relabeling was on; otw [(id: None)]
    """
    if not relabeling:
        raise NotImplementedError

    def get_answer_text(ids, q_id, answer):
        answer_text = None
        for i in ids:
            if i[0] == q_id:
                try:
                    answer_text = i[1][answer]
                except KeyError:
                    print()
        assert answer_text is not None
        return answer_text

    def prepare_answer(txt):
        assert txt.startswith("@entity")
        return txt[len("@entity"):].replace("_", " ")

    qid_ans_lst = list(zip((i[0] for i in ids), (inv_entity_dict[index] for index in preds)))
    # answers are anonymized under relabeling
    if relabeling:
        qid_ans = {q_id: prepare_answer(get_answer_text(ids, q_id, answer)) for q_id, answer in qid_ans_lst}

    return qid_ans


def to_output_att(ids, relabeling, att_weights, qs, inv_word_dict):
    """
    :param ids: [(id: {@entity0: @entityAnswer})] if relabeling was on; otw [(id: None)]
    :param att_weights: [[(w_id, att), ...], ...]
    """
    if not relabeling:
        raise NotImplementedError

    qid_p_atts = {}
    assert len(att_weights) == len(qs)
    for n, instance in enumerate(att_weights):
        deindexed_x1 = []
        for i, att in instance:
            if i == 0:
                deindexed_x1.append(("UNK", float(att)))
            else:
                deindexed_x1.append((inv_word_dict[i], float(att)))
        final_deindexed_x1 = []
        if relabeling:
            for w, att in deindexed_x1:
                if w in ids[n][1]:
                    final_deindexed_x1.append((ids[n][1][w], att))
                else:
                    final_deindexed_x1.append((w, att))
        else:
            final_deindexed_x1 = deindexed_x1
        deindexed_x2 = []
        for i in qs[n]:
            if i == 0:
                deindexed_x2.append("UNK")
            else:
                deindexed_x2.append(inv_word_dict[i])
        final_deindexed_x2 = []
        if relabeling:
            for w in deindexed_x2:
                if w in ids[n][1]:
                    final_deindexed_x2.append(ids[n][1][w])
                else:
                    final_deindexed_x2.append(w)
        else:
            final_deindexed_x2 = deindexed_x2
        qid_p_atts[ids[n][0]] = {"d_att": final_deindexed_x1, "q": final_deindexed_x2}

    return qid_p_atts


def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')

    if args.debug:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, 5, relabeling=args.relabeling,
                                         remove_notfound=args.remove_notfound)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, 100, relabeling=args.relabeling,
                                       remove_notfound=args.remove_notfound)
    elif args.test_only:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_cnn_data(args.train_file, relabeling=args.relabeling)  # docs, qs, ans
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling,
                                       remove_notfound=args.remove_notfound)
    elif args.cnn_train:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_cnn_data(args.train_file, relabeling=args.relabeling)  # docs, qs, ans
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_cnn_data(args.dev_file, args.max_dev, relabeling=args.relabeling)
    else:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, relabeling=args.relabeling,
                                         remove_notfound=args.remove_notfound)  # docs, qs, ans
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling,
                                       remove_notfound=args.remove_notfound)

    args.num_train = len(train_examples[0])
    args.num_dev = len(dev_examples[0])

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dict = utils.build_dict(train_examples[0] + train_examples[1],  # + dev_examples[0] + dev_examples[1],
                                 max_words=args.max_words)  # docs+qs
    entity_markers = list(set([w for w in word_dict.keys()
                              if w.startswith('@entity')] + train_examples[2]))
    entity_markers = ['<unk_entity>'] + entity_markers
    entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    inv_entity_dict = {index: w for w, index in entity_dict.items()}
    assert len(entity_dict) == len(inv_entity_dict)
    logging.info('Entity markers: %d' % len(entity_dict))
    args.num_labels = len(entity_dict)

    logging.info('-' * 50)
    # Load embedding file
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.info('Compile functions..')
    train_fn, test_fn, params = build_fn(args, embeddings)
    logging.info('Done.')

    logging.info('-' * 50)
    logging.info(args)

    logging.info('-' * 50)
    logging.info('Intial test..')
    dev_x1, dev_x2, dev_l, dev_y, dev_ids = utils.vectorize(dev_examples, word_dict, entity_dict,
                                                   remove_notfound=args.remove_notfound,
                                                   relabeling=args.relabeling)
    if dev_ids is not None:
        assert len(dev_y) == len(dev_ids)
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
    dev_acc, dev_preds, dev_att_ws, dev_qs = eval_acc(test_fn, all_dev, args.att_output)

    if dev_ids is not None:
        assert len(dev_ids) == len(dev_preds) == len(dev_y)
    if args.att_output:
        assert dev_att_ws
        assert len(dev_preds) == len(dev_att_ws)
    if dev_ids is not None:
        dev_preds_data = to_output_preds(dev_ids, dev_preds, inv_entity_dict, args.relabeling)
        dev_att_data = to_output_att(dev_ids, args.relabeling, dev_att_ws, dev_qs, {index: w for w, index in word_dict.items()})
    logging.info('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc

    if args.log_file is not None:
        assert args.log_file.endswith(".log")
        run_name = args.log_file[:args.log_file.find(".log")]
        if dev_ids is not None:
            preds_file_name = run_name + ".preds"
            utils.write_preds(dev_preds_data, preds_file_name)
            utils.external_eval(preds_file_name, run_name + ".preds.scores", args.dev_file)
            if args.att_output:
                import json
                def save_json(obj, filename):
                    with open(filename, "w") as out:
                        json.dump(obj, out, separators=(',', ':'))
                save_json(dev_ids, run_name + ".preds.ids")
                preds_att_file_name = preds_file_name + ".att"
                utils.write_att(dev_att_data, preds_att_file_name)

    if args.test_only:
        return

    utils.save_params(args.model_file, params, epoch=0, n_updates=0)

    # Training
    logging.info('-' * 50)
    logging.info('Start training..')
    train_x1, train_x2, train_l, train_y, train_ids = utils.vectorize(train_examples, word_dict, entity_dict,
                                                           remove_notfound=args.remove_notfound,
                                                           relabeling=args.relabeling)
    assert len(train_x1) == args.num_train
    start_time = time.time()
    n_updates = 0
    train_accs = []
    dev_accs = []
    all_train = gen_examples(train_x1, train_x2, train_l, train_y, args.batch_size)
    for epoch in range(args.num_epoches):
        np.random.shuffle(all_train)
        for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y) in enumerate(all_train):
            logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
            train_loss = train_fn(mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y)
            logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' %
                         (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1

            if n_updates % args.eval_iter == 0:
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                                            [train_x2[k] for k in samples],
                                            train_l[samples],
                                            [train_y[k] for k in samples],
                                            args.batch_size)
                train_acc, train_preds, train_att_ws, train_qs = eval_acc(test_fn, sample_train, args.att_output)
                train_accs.append(train_acc)
                logging.info('Train accuracy: %.2f %%' % train_acc)
                dev_acc, dev_preds, dev_att_ws, dev_qs = eval_acc(test_fn, all_dev, args.att_output)
                dev_accs.append(dev_acc)
                logging.info('Dev accuracy: %.2f %%' % dev_acc)
                utils.update_plot(args.eval_iter, train_accs, dev_accs, file_name=args.log_file + ".html")
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, dev_acc))
                    if args.log_file is not None:
                        #utils.save_params(args.model_file, params, epoch=epoch, n_updates=n_updates)
                        utils.save_params(run_name + ".model", params, epoch=epoch, n_updates=n_updates)
                        if dev_ids is not None:
                            dev_preds_data = to_output_preds(dev_ids, dev_preds, inv_entity_dict, args.relabeling)
                            utils.write_preds(dev_preds_data, preds_file_name)
                            utils.external_eval(preds_file_name, run_name + ".preds.scores", args.dev_file)


if __name__ == '__main__':
    args = config.get_args()
    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))

    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    if args.rnn_type == 'lstm':
        args.rnn_layer = lasagne.layers.LSTMLayer
    elif args.rnn_type == 'gru':
        args.rnn_layer = lasagne.layers.GRULayer
    else:
        raise NotImplementedError('rnn_type = %s' % args.rnn_type)

    if args.embedding_file is not None:
        dim = utils.get_dim(args.embedding_file)
        if (args.embedding_size is not None) and (args.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                             (args.embedding_size, args.embedding_file, dim))
        args.embedding_size = dim
    elif args.embedding_size is None:
        raise RuntimeError('Either embedding_file or embedding_size needs to be specified.')

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))
    main(args)
