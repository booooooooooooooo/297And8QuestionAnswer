import tensorflow as tf
import os
from tqdm import tqdm
import datetime
import random
import numpy as np
import json

from evaluate_v_1_1 import f1_score, exact_match_score
from util_data import *
from Encoder.EncoderPreprocess import EncoderPreprocess
from Encoder.EncoderMatch import EncoderMatch
from Decoder.DecoderAnsPtr import DecoderAnsPtr





class Model:
    def __init__(self, embed_matrix, config):
        self.embed_matrix = embed_matrix
        self.config = config
        #Train, valid and test data must be consistent on these parameters.
        self.pass_max_length = config["pass_max_length"]
        self.ques_max_length = config["ques_max_length"]
        self.embed_size = config["embed_size"]
        #not related to data.
        self.num_units = config["num_units"]
        self.clip_norm = config["clip_norm"]
        self.lr = config["lr"]
        self.regularizer = tf.contrib.layers.l2_regularizer(config["reg_scale"])
        self.n_epoch = config["n_epoch"]
        self.arch = config["arch"]
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
        self.passage_mask = tf.placeholder(tf.float32, shape = (None, pass_max_length), name = "passage_mask_placeholder")
        self.ques = tf.placeholder(tf.int32, shape = (None, ques_max_length), name = "question_placeholder")
        self.ques_mask = tf.placeholder(tf.float32, shape = (None, ques_max_length), name = "question_mask_placeholder")
        self.answer_s = tf.placeholder(tf.int32, (None,), name = "answer_start")
        self.answer_e = tf.placeholder(tf.int32, (None,), name = "answer_end")


    def add_predicted_dist(self):
        # print self.arch
        if self.arch == "match_simple":
            H_p, H_q = EncoderPreprocess(self.embed_matrix, self.passage, self.passage_mask, self.ques, self.ques_mask, self.num_units).encode()
            H_r = EncoderMatch(H_q, self.ques_mask, H_p, self.passage_mask, 2 * self.num_units, self.num_units, "simple").encode()
            beta_s, beta_e = DecoderAnsPtr(H_r, self.passage_mask, 2 * self.num_units).decode()
        elif self.arch == "match":
            H_p, H_q = EncoderPreprocess(self.embed_matrix, self.passage, self.passage_mask, self.ques, self.ques_mask, self.num_units).encode()
            H_r = EncoderMatch(H_q, self.ques_mask, H_p, self.passage_mask, 2 * self.num_units, self.num_units, "general").encode()
            beta_s, beta_e = DecoderAnsPtr(H_r, self.passage_mask, 2 * self.num_units).decode()
        elif self.arch == "r_net":
            H_p, H_q = EncoderPreprocess(self.embed_matrix, self.passage, self.passage_mask, self.ques, self.ques_mask, self.num_units).encode()
            with tf.variable_scope("match_p_q"):
                H_r = EncoderMatch(H_q, self.ques_mask, H_p, self.passage_mask, 2 * self.num_units, self.num_units, "gated").encode()
            with tf.variable_scope("match_p_p"):
                H_t = EncoderMatch(H_r, self.passage_mask, H_r, self.passage_mask, 2 * self.num_units, self.num_units, "general").encode()
            beta_s, beta_e = DecoderAnsPtr(H_t, self.passage_mask, 2 * self.num_units).decode()
        elif self.arch == "r_net_iter":
            H_p, H_q = EncoderPreprocess(self.embed_matrix, self.passage, self.passage_mask, self.ques, self.ques_mask, self.num_units).encode()
            with tf.variable_scope("match_p_q"):
                H_r = EncoderMatch(H_q, self.ques_mask, H_p, self.passage_mask, 2 * self.num_units, self.num_units, "gated").encode()
            with tf.variable_scope("match_p_p_0"):
                H_t = EncoderMatch(H_r, self.passage_mask, H_r, self.passage_mask, 2 * self.num_units, self.num_units, "general").encode()
            with tf.variable_scope("match_p_p_1"):
                H_u = EncoderMatch(H_t, self.passage_mask, H_t, self.passage_mask, 2 * self.num_units, self.num_units, "general").encode()
            beta_s, beta_e = DecoderAnsPtr(H_u, self.passage_mask, 2 * self.num_units).decode()
        else:
            raise ValueError('Architecture should be match_simple, match, r_net or r_net_iter')

        self.beta_s, self.beta_e = tf.identity(beta_s, name="beta_s"), tf.identity(beta_e, name="beta_e")

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
        gradients, grad_norm  = tf.clip_by_global_norm(gradients, self.clip_norm)
        train_op = optimizer_func.apply_gradients(zip(gradients, variables))#second part of minimize()

        self.train_op = train_op
        self.grad_norm = grad_norm

    '''Functions below are used for train and validation'''
    def predict(self, sess, passage, passage_mask, ques ,ques_mask):
        dist_s, dist_e = sess.run([self.beta_s, self.beta_e], {self.passage: passage,
                                                               self.passage_mask: passage_mask,
                                                               self.ques: ques,
                                                               self.ques_mask: ques_mask})
        idx_s = np.argmax(dist_s, axis=1)
        idx_e = np.argmax(dist_e, axis=1)

        return idx_s, idx_e
    def evaluate(self, sess, data_tuple, batch_size, sample_size):
        passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text, voc = data_tuple
        bound = min(sample_size, len(passage))
        passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text = passage[0: bound],\
                                                                                  passage_mask[0: bound],\
                                                                                  ques[0: bound],\
                                                                                  ques_mask[0: bound],\
                                                                                  answer_s[0: bound],\
                                                                                  answer_e[0: bound],\
                                                                                  answer_text[0: bound]



        predict_text = []
        f1 = 0.0
        em = 0.0
        loss = 0.0
        iters = sample_size / batch_size if sample_size % batch_size == 0 else sample_size / batch_size + 1
        for i in tqdm(xrange(iters), desc = "Evaluating......." ):
            l, r = i * batch_size, min((i + 1) * batch_size, len(passage))

            idx_s, idx_e = self.predict(sess, passage[l : r], passage_mask[l : r], ques[l : r] ,ques_mask[l : r])
            b_predict_text = predict_ans_text(idx_s, idx_e, passage[l : r], voc)
            predict_text = predict_text + b_predict_text
            for j in xrange(len(answer_text[l : r])):
                f1 += f1_score(predict_text[j].decode('utf8'), answer_text[j].decode('utf8'))
                em += exact_match_score(predict_text[j].decode('utf8'), answer_text[j].decode('utf8'))
            b_loss = sess.run(self.loss, {self.passage: passage[l : r],
                                        self.passage_mask: passage_mask[l : r],
                                        self.ques: ques[l : r],
                                        self.ques_mask: ques_mask[l : r],
                                        self.answer_s: answer_s[l : r],
                                        self.answer_e: answer_e[l : r]})
            loss += b_loss

        f1 /= len(passage)
        em /= len(passage)
        loss /= iters

        return loss, f1, em, predict_text

    def fit(self, sess, train_data, valid_data, test_data, batch_size, sample_size, dir_output):
        if not os.path.isdir(dir_output):
            os.makedirs(dir_output)
        if not os.path.isdir(os.path.join(dir_output, "graphes/")):
            os.makedirs(os.path.join(dir_output, "graphes/"))

        #Just in case. sess para should be already initialized
        print "Start intializing graph"
        sess.run(tf.global_variables_initializer())#Initilizing after making train_op
        print "Finish intializing graph"

        stat = {}
        stat["config"] = self.config
        stat["train_stat"] = []
        for epoch in xrange(self.n_epoch):
            train_stat = []
            passage, passage_mask, ques, ques_mask, answer_s, answer_e, answer_text, voc = train_data
            batches = get_batches(passage, passage_mask, ques, ques_mask, answer_s, answer_e, batch_size)
            for num in xrange(len(batches)):
                b_passage, b_passage_mask, b_ques, b_ques_mask, b_answer_s, b_answer_e = batches[num]
                _ , batch_loss, batch_grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], {self.passage: b_passage,
                                                                       self.passage_mask: b_passage_mask,
                                                                       self.ques: b_ques,
                                                                       self.ques_mask: b_ques_mask,
                                                                       self.answer_s: b_answer_s,
                                                                       self.answer_e: b_answer_e})


                print "epoch: {}, batch: {} / {}, batch_loss: {}, batch_grad_norm: {}".format(epoch, num, len(batches), batch_loss, batch_grad_norm)
                if num % 100 == 0:
                    graph_file = os.path.join(dir_output, "graphes/", datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S"))
                    tf.train.Saver().save(sess, graph_file)
                    train_loss, train_f1, train_em, _ = self.evaluate(sess, train_data, batch_size, sample_size)
                    valid_loss, valid_f1, valid_em, _ = self.evaluate(sess, valid_data, batch_size, sample_size)
                    batch_stat = {"epoch": str(epoch), "batch": str(num), "batch_loss" : str(batch_loss), \
                                  "batch_grad_norm": str(batch_grad_norm),\
                                  "sample_size": str(sample_size), "train_loss": str(train_loss),\
                                  "train_f1" : str(train_f1), "train_em": str(train_em), \
                                  "valid_loss": str(valid_loss), "valid_f1": str(valid_f1), \
                                  "valid_em": str(valid_em) , "graph_file" : graph_file}
                    train_stat.append(batch_stat)
                    print "================"
                    print "validation_sample_size: {}".format(sample_size)
                    print "Sample train_loss: {}, train_f1 : {}, train_em : {}".format( train_loss, train_f1, train_em)
                    print "Sample valid_loss: {}, valid_f1: {}, valid_em: {}".format(valid_loss, valid_f1, valid_em)
                    print "================"
                    break

            stat["train_stat"] += train_stat

        print "Finish trainning! Congs!"

        print "=================="
        #find best graph
        print "Searching for best graph"
        best_graph = ""
        best_score = -1.0
        for batch_stat in stat["train_stat"]:
            cur_score = (float(batch_stat["valid_f1"]) + float(batch_stat["valid_em"]) )/ 2.0
            if cur_score >= best_score:
                best_graph = batch_stat["graph_file"]
                best_score = cur_score
        print "Have found best graph {}".format(best_graph)
        stat["best_graph"] = {"graph": best_graph, "score": str(best_score)}

        print "=================="
        print "Start testing!"

        #recover best graph
        print "Recovering best graph"
        saver = tf.train.import_meta_graph(best_graph + '.meta')
        saver.restore(sess, best_graph)
        print "Have Recovered best graph"
        #get predictions and scores
        print "Start evaluating!"
        loss, f1, em, predict_test_answers = self.evaluate(sess, test_data, batch_size, len(test_data[0]))

        predict_test_answers_file = os.path.join(dir_output, "predict_test_answers-{}".format(datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S")))
        print "Finish evaluating!"
        #write to disk and IO
        print "Writing predict test answers to disk"
        with open(predict_test_answers_file, 'w') as f:
            for ans in predict_test_answers:
                f.write(ans + "\n")
        print "Finish writing!"
        print "Test loss : {}".format(loss)
        print "Test f1: {}".format(f1)
        print "Test em: {}".format(em)
        print "Test predicted answer saved at {}".format(predict_test_answers_file)
        stat["test_stat"] = {"loss": str(loss), "f1": str(f1), "em": str(em), "predicted_ans_file": predict_test_answers_file}
        print "Finish tesing! Congs!"
        print "=================="


        with open(os.path.join(dir_output, "Stat-{}".format(datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%S"))), 'w') as f:
            f.write(json.dumps(stat))
