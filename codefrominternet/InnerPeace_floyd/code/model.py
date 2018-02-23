import tensorflow as tf
import os
from tqdm import tqdm
import datetime
import random
import numpy as np


from evaluate import exact_match_score, f1_score


class MatchGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size, H_q, ques_mask):
        '''
        input_size: size of encoded passage vector
        state_size = size of this cell's hidden units
        '''
        self._input_size = input_size
        self._state_size = state_size
        self.H_q = H_q
        self.ques_mask = ques_mask

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        '''
        para:
            inputs: (batch_size, _input_size)
            state: (batch_size, _state_size)
        '''
        scope = scope or type(self).__name__
        with tf.variable_scope(scope):
            _input_size = self._input_size
            _state_size = self._state_size
            H_q = self.H_q
            ques_mask = self.ques_mask
            batch_size = tf.shape(H_q)[0]
            ques_max_length = tf.shape(H_q)[1]

            init = tf.contrib.layers.xavier_initializer()
            #attention
            W_q = tf.get_variable("W_q", initializer = init, shape = (_input_size, _input_size), dtype = tf.float32)
            W_p = tf.get_variable("W_p", initializer = init, shape = (_input_size, _input_size), dtype = tf.float32)
            W_r_0 = tf.get_variable("W_r_0", initializer = init, shape = (_state_size, _input_size), dtype = tf.float32)
            b_p = tf.get_variable("b_p", initializer = init, shape = (_input_size, ), dtype = tf.float32)
            w = tf.get_variable("w", initializer = init, shape = (_input_size, ), dtype = tf.float32)
            b = tf.get_variable("b", initializer = init, shape = (), dtype = tf.float32)

            G = tf.tanh(tf.matmul(H_q, tf.tile(tf.expand_dims(W_q, axis = [0]), [batch_size, 1, 1]))
                    + tf.expand_dims(tf.matmul(inputs, W_p) + tf.matmul(state, W_r_0) + b_p, [1]))#(batch_size, ques_max_length, _input_size)
            alpha = tf.reshape(tf.matmul(G, tf.tile(tf.reshape(w, [1, _input_size, 1]), [batch_size, 1, 1])), [batch_size, -1])#(batch_size, ques_max_length)
            #TODO:use 10e-1 to mask alpha before appling softmax ??
            alpha = tf.nn.softmax(alpha) * ques_mask #(batch_size, ques_max_length)
            att_v = tf.reshape(tf.matmul(tf.expand_dims(alpha, [1]), H_q), [batch_size, _input_size])#(batch_size, _input_size)
            z = tf.concat([inputs, att_v], 1)#(batch_size, 2 * _input_size)

            #gru cell
            #TODO: change 2 * _input_size to _input_size + tf.shape(inputs)[1] ??
            W_z = tf.get_variable('W_z', (2 * _input_size, _state_size), tf.float32, init)
            U_z = tf.get_variable('U_z', (_state_size, _state_size), tf.float32, init)
            b_z = tf.get_variable('b_z', (_state_size,), tf.float32, init)

            W_r_1 = tf.get_variable('W_r_1', (2 * _input_size, _state_size), tf.float32, init)
            U_r = tf.get_variable('U_r', (_state_size, _state_size), tf.float32, init)
            b_r = tf.get_variable('b_r', (_state_size,), tf.float32, init)


            W_h = tf.get_variable('W_h', (2 * _input_size, _state_size), tf.float32, init)
            U_h = tf.get_variable('U_h', (_state_size, _state_size), tf.float32, init)
            b_h = tf.get_variable('b_h', (_state_size,), tf.float32, init)


            z_gru = tf.nn.sigmoid(tf.matmul(z, W_z) + tf.matmul(state, U_z) + b_z)
            r_gru = tf.nn.sigmoid(tf.matmul(z, W_r_1) + tf.matmul(state, U_r) + b_r)
            new_state_wave = tf.nn.tanh(tf.matmul(z, W_h) + tf.matmul(r_gru * state, U_h) + b_h)
            new_state = z_gru * state + (1 - z_gru) * new_state_wave
            output = new_state

        return output, new_state


class Model:
    def __init__(self, embed_matrix, pass_max_length, ques_max_length, embed_size, num_units, clip_norm, lr, n_epoch, reg_scale):
        #Train, valid and test data must be consistent on these parameters.
        self.embed_matrix = embed_matrix
        self.pass_max_length = pass_max_length
        self.ques_max_length = ques_max_length
        self.embed_size = embed_size
        #not related to data.
        self.num_units = num_units
        self.clip_norm = clip_norm
        self.lr = lr
        self.regularizer = tf.contrib.layers.l2_regularizer(reg_scale)
        self.n_epoch = n_epoch
        #build the graph
        self.add_placeholder()
        self.add_predicted_dist()
        self.add_loss_function()
        self.add_train_op()

    def add_placeholder(self):
        pass_max_length = self.pass_max_length
        ques_max_length = self.ques_max_length
        embed_size = self.embed_size

        self.passage = tf.placeholder(tf.int32, shape = (None, pass_max_length), name = "passage_placeholder")
        self.passage_mask = tf.placeholder(tf.float32, shape = (None, pass_max_length), name = "passage_sequence_length_placeholder")
        self.ques = tf.placeholder(tf.int32, shape = (None, ques_max_length), name = "question_placeholder")
        self.ques_mask = tf.placeholder(tf.float32, shape = (None, ques_max_length), name = "question_sequence_length_placeholder")
        self.answer_s = tf.placeholder(tf.int32, (None,), name = "answer_start")
        self.answer_e = tf.placeholder(tf.int32, (None,), name = "answer_end")


    def encode(self, embed_matrix, passage, passage_mask, ques, ques_mask, num_units, regularizer):
        '''
        return:
            H_p: (batch_size, pass_max_length, 2 * num_units)
            H_q: (batch_size, ques_max_length, 2 * num_units)
        '''
        with tf.variable_scope("embedding"):
            passage_embed = tf.nn.embedding_lookup(embed_matrix, passage)
            ques_embed = tf.nn.embedding_lookup(embed_matrix, ques)

        with tf.variable_scope("encode_preprocess"):
            lstm_pre_fw = tf.contrib.rnn.BasicLSTMCell(num_units)
            lstm_pre_bw = tf.contrib.rnn.BasicLSTMCell(num_units)

            H_p_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, passage_embed,
                                       sequence_length = tf.reduce_sum(tf.cast(passage_mask, tf.int32), axis=1),
                                       dtype=tf.float32)
            H_q_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_pre_fw, lstm_pre_bw, ques_embed,
                                       sequence_length = tf.reduce_sum(tf.cast(ques_mask, tf.int32), axis=1),
                                       dtype=tf.float32)
            H_p = tf.concat(H_p_pair, 2)
            H_q = tf.concat(H_q_pair, 2)


        with tf.variable_scope("encode_match"):
            lstm_gru_fw = MatchGRUCell(2 * num_units, 2 * num_units, H_q, ques_mask)
            lstm_gru_bw = MatchGRUCell(2 * num_units, 2 * num_units, H_q, ques_mask)


            H_r_pair, _ = tf.nn.bidirectional_dynamic_rnn(lstm_gru_fw,
                                                         lstm_gru_bw,
                                                         H_p,
                                                         sequence_length=tf.reduce_sum(tf.cast(passage_mask, tf.int32), axis=1),
                                                         dtype=tf.float32)
            H_r = tf.concat(H_r_pair, 2)

        return H_r


    def decode(self, H_r, pass_max_length, passage_mask, num_units, regularizer):
        '''
        paras
            H_r:        (batch_size, pass_max_length, 2 * num_units)
        return
            dist:       (batch_size, 2, pass_max_length)
        '''
        batch_size = tf.shape(H_r)[0]

        init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("decode_ans_ptr"):
            V = tf.get_variable("V", shape = (4 * num_units, 2 * num_units), dtype = tf.float32, initializer = init)
            W_a = tf.get_variable("W_a", shape = (4 * num_units, 2 * num_units), dtype = tf.float32, initializer = init)
            b_a = tf.get_variable("b_a", shape = (2 * num_units, ), dtype = tf.float32, initializer = init)
            v = tf.get_variable("v", shape = (2 * num_units, ), dtype = tf.float32, initializer = init)
            c = tf.get_variable("c", shape = (), dtype = tf.float32, initializer = init)


            F_s = tf.tanh( tf.matmul(H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                           + b_a)#(batch_size, pass_max_length, num_units)
            beta_s = tf.squeeze(tf.matmul(F_s,
                                          tf.tile(tf.reshape(v, [1, -1, 1]),
                                                  [batch_size, 1, 1])))#(batch_size, pass_max_length)





            prob_s = tf.nn.softmax(beta_s) * passage_mask#(batch_size, pass_max_length)
            h_a = tf.squeeze(tf.matmul(tf.expand_dims(prob_s, [1]), H_r))#(batch_size, 2 * num_units)
            F_e = tf.tanh( tf.matmul(H_r, tf.tile(tf.expand_dims(V, [0]), [batch_size, 1, 1]))
                        +  tf.expand_dims(tf.matmul(h_a, W_a) + b_a , [1]))#(batch_size, pass_max_length, num_units)
            beta_e = tf.squeeze(tf.matmul(F_e, tf.tile(tf.reshape(v, [1, -1, 1]), [batch_size, 1, 1])))#(batch_size, pass_max_length)




        return beta_s, beta_e

        # #TODO: why appling softmax makes dizaster?
        # return tf.nn.softmax(beta_s) , tf.nn.softmax(beta_e)


    def add_predicted_dist(self):
        H_r = self.encode(self.embed_matrix, self.passage, self.passage_mask, self.ques, self.ques_mask, self.num_units, self.regularizer)
        self.beta_s, self.beta_e = self.decode(H_r, self.pass_max_length, self.passage_mask, self.num_units, self.regularizer)

    def add_loss_function(self):
        loss_s = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=self.beta_s, labels=self.answer_s))
        loss_e = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=self.beta_e, labels=self.answer_e)
                                    )
        reg_losses = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_losses)

        self.loss = (loss_s + loss_e) / 2.0 + reg_term


    def add_train_op(self):
        optimizer_func = tf.train.AdamOptimizer(self.lr)

        gradients, variables = zip(*optimizer_func.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        train_op = optimizer_func.apply_gradients(zip(gradients, variables))#second part of minimize()

        self.train_op = train_op


    #for debugging
    def decode_debug(self, session, context, question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        context_data = [x[0] for x in context]
        context_masks = [x[1] for x in context]
        question_data = [x[0] for x in question]
        question_masks = [x[1] for x in question]

        input_feed = {self.passage: context_data,
                      self.passage_mask: context_masks,
                      self.ques: question_data,
                      self.ques_mask: question_masks,
                      }

        output_feed = [self.beta_s, self.beta_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, context, question):

        yp, yp2 = self.decode_debug(session, context, question)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return a_s, a_e, yp, yp2

    def evaluate_answer(self, session, dataset, answers, rev_vocab,
                        set_name='val', training=False, log=False,
                        sample=(100, 100), sendin=None, ensemble=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        if not isinstance(rev_vocab, np.ndarray):
            rev_vocab = np.array(rev_vocab)

        if not isinstance(sample, tuple):
            sample = (sample, sample)

        input_batch_size = 100

        if training:
            train_context = dataset['train_context'][:sample[0]]
            train_question = dataset['train_question'][:sample[0]]
            train_answer = answers['raw_train_answer'][:sample[0]]
            train_len = len(train_context)

            if sendin and len(sendin) > 2:
                train_a_s, train_a_e = sendin[0:2]
            else:
                train_a_e = np.array([], dtype=np.int32)
                train_a_s = np.array([], dtype=np.int32)

                for i in tqdm(range(train_len // input_batch_size), desc='trianing set'):
                    # sys.stdout.write('>>> %d / %d \r'%(i, train_len // input_batch_size))
                    # sys.stdout.flush()
                    train_as, train_ae, yp, yp2 = self.answer(session,
                                                     train_context[i * input_batch_size:(i + 1) * input_batch_size],
                                                     train_question[i * input_batch_size:(i + 1) * input_batch_size])
                    train_a_s = np.concatenate((train_a_s, train_as), axis=0)
                    train_a_e = np.concatenate((train_a_e, train_ae), axis=0)
                    # print(yp[0])
                    # print(yp2[0])
                    # print("")

            tf1 = 0.
            tem = 0.
            for i, con in enumerate(train_context):
                # #commented by bo
                # sys.stdout.write('>>> %d / %d \r' % (i, train_len))
                # sys.stdout.flush()
                prediction_ids = con[0][train_a_s[i]: train_a_e[i] + 1]
                prediction = rev_vocab[prediction_ids]
                prediction = ' '.join(prediction)
                # if i < 10:
                #     print('context: {}'.format(rev_vocab[con[0]]))
                #     print('prediction: {}'.format( prediction))
                #     print(' g-truth:   {}'.format( train_answer[i]))
                #     print('f1_score: {}'.format(f1_score(prediction, train_answer[i])))

                tf1 += f1_score(prediction, train_answer[i])
                tem += exact_match_score(prediction, train_answer[i])

            print("Training set ==> F1: {}, EM: {}, for {} samples".
                             format(tf1 / train_len, tem / train_len, train_len))

        # it was set to 1.0
        f1 = 0.0
        em = 0.0
        val_context = dataset[set_name + '_context'][:sample[1]]
        val_question = dataset[set_name + '_question'][:sample[1]]
        # ['Corpus Juris Canonici', 'the Northside', 'Naples', ...]
        val_answer = answers['raw_val_answer'][:sample[1]]

        val_len = len(val_context)
        # logging.info('calculating the validation set predictions.')

        if sendin and len(sendin) > 2:
            val_a_s, val_a_e = sendin[-2:]
        elif sendin:
            val_a_s, val_a_e = sendin
        else:
            val_a_s = np.array([], dtype=np.int32)
            val_a_e = np.array([], dtype=np.int32)
            for i in tqdm(range(val_len // input_batch_size), desc='validation   '):
                # sys.stdout.write('>>> %d / %d \r'%(i, val_len // input_batch_size))
                # sys.stdout.flush()
                a_s, a_e, yp, yp2 = self.answer(session, val_context[i * input_batch_size:(i + 1) * input_batch_size],
                                       val_question[i * input_batch_size:(i + 1) * input_batch_size])
                val_a_s = np.concatenate((val_a_s, a_s), axis=0)
                val_a_e = np.concatenate((val_a_e, a_e), axis=0)

        # logging.info('getting scores of dev set.')
        for i, con in enumerate(val_context):
            # sys.stdout.write('>>> %d / %d \r' % (i, val_len))
            # sys.stdout.flush()
            prediction_ids = con[0][val_a_s[i]: val_a_e[i] + 1]
            prediction = rev_vocab[prediction_ids]
            prediction = ' '.join(prediction)
            # if i < 10:
                # print('context : {}'.format(' '.join(rev_vocab[con[0]])))
                # print('question: {}'.format(' '.join(rev_vocab[val_question[i][0]])))
                # print('prediction: {}'.format( prediction))
                # print(' g-truth:   {}'.format( val_answer[i]))
                # print('f1_score: {}'.format(f1_score(prediction, val_answer[i])))
            f1 += f1_score(prediction, val_answer[i])
            em += exact_match_score(prediction, val_answer[i])

        print("Validation   ==> F1: {}, EM: {}, for {} samples".
                         format(f1 / val_len, em / val_len, val_len))
        # pdb.set_trace()

        if ensemble and training:
            return train_a_s, train_a_e, val_a_s, val_a_e
        elif ensemble:
            return val_a_s, val_a_e
        # else:
        #    return , train_a_e, val_a_s, val_a_e
        else:
            return tf1 / train_len, tem / train_len, f1 / val_len, em / val_len

    def train(self, lr, session, dataset, answers, train_dir, debug_num=0, raw_answers=None,
              rev_vocab=None):

        train_context = np.array(dataset['train_context'])
        train_question = np.array(dataset['train_question'])
        train_answer = np.array(answers['train_answer'])


        num_example = len(train_answer)
        shuffle_list = np.arange(num_example)


        batch_size = 32
        batch_num = int(num_example / batch_size)


        for ep in xrange(1):#only run 1 epoch
            # TODO: add random shuffle.
            np.random.shuffle(shuffle_list)
            train_context = train_context[shuffle_list]
            train_question = train_question[shuffle_list]
            train_answer = train_answer[shuffle_list]

            ep_loss = 0.
            for it in xrange(batch_num):

                context = train_context[it * batch_size: (it + 1) * batch_size]
                question = train_question[it * batch_size: (it + 1) * batch_size]
                answer = train_answer[it * batch_size: (it + 1) * batch_size]


                context_data = [x[0] for x in context]
                context_masks = [x[1] for x in context]
                question_data = [x[0] for x in question]
                question_masks = [x[1] for x in question]
                answer_start = [x[0] for x in answer]
                answer_end = [x[1] for x in answer]

                input_feed = {self.passage: context_data,
                              self.passage_mask: context_masks,
                              self.ques: question_data,
                              self.ques_mask: question_masks,
                              self.answer_s: answer_start,
                              self.answer_e: answer_end,
                              }

                loss, _ = session.run([self.loss, self.train_op], input_feed)

                ep_loss += loss

                print( "{}/{}".format(it, batch_num) )
                print("loss {}".format(loss))

                if it % 20 == 0:
                    self.evaluate_answer(session, dataset, raw_answers, rev_vocab,
                                                            training=True, log=True, sample=100)
    def save_predictions(self, session, dataset, answers, rev_vocab,set_name='val'):
        print("Making predictions on validation dataset")

        if not isinstance(rev_vocab, np.ndarray):
            rev_vocab = np.array(rev_vocab)

        input_batch_size = 100

        print("Getting validation context")
        val_context = dataset[set_name + '_context']
        print("Getting validation question")
        val_question = dataset[set_name + '_question']
        # ['Corpus Juris Canonici', 'the Northside', 'Naples', ...]
        print("Getting validation answer")
        val_answer = answers['raw_val_answer']

        val_len = len(val_context)
        # logging.info('calculating the validation set predictions.')

        val_a_s = np.array([], dtype=np.int32)
        val_a_e = np.array([], dtype=np.int32)

        batches = min(10, val_len // input_batch_size)
        print("Answering validation questions of beginning {} batches".format(batches))
        for i in range(batches):
            # sys.stdout.write('>>> %d / %d \r'%(i, val_len // input_batch_size))
            # sys.stdout.flush()
            print("batch {}".format(i))
            a_s, a_e, yp, yp2 = self.answer(session, val_context[i * input_batch_size:(i + 1) * input_batch_size],
                                   val_question[i * input_batch_size:(i + 1) * input_batch_size])
            val_a_s = np.concatenate((val_a_s, a_s), axis=0)
            val_a_e = np.concatenate((val_a_e, a_e), axis=0)

        predictions = []
        # logging.info('getting scores of dev set.')
        for i in xrange(input_batch_size * batches):
            print("Prediction {}".format(i))
            # sys.stdout.write('>>> %d / %d \r' % (i, val_len))
            # sys.stdout.flush()
            con = val_context[i]
            prediction_ids = con[0][val_a_s[i]: val_a_e[i] + 1]
            prediction = rev_vocab[prediction_ids]
            # print(prediction)
            prediction = ' '.join(prediction)
            predictions.append(prediction)
        print("Saving predictions")
        with open(os.path.join("/output", "valid_predictions.txt"), 'w') as f:
            for prediction in predictions:
                f.write(prediction + '\n')
