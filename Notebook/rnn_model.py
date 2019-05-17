"""
An exampe to show how to use rnn with tf
"""

import tensorflow as tf
import numpy as np
import time
import os


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # set all the entries to zero except the top N entries
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    return np.random.choice(vocab_size, 1, p=p)[0]


class CharRNN:
    def __init__(self, n_classes, batch_size=64, num_steps=50,
                 lstm_size=128, n_layers=2, lr=1e-3, grad_clip=5, sampling=False,
                 train_keep_prob=0.5, use_embedding=False, embedding_size=128):

        if sampling is True:
            batch_size, num_steps = 1, 1

        self.n_classes = n_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.n_layers = n_layers
        self.lr = lr
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self._build_input()
        self._build_net()
        self._build_loss()
        self._build_optimizer()
        self.saver = tf.train.Saver()

    def _build_input(self):
        with tf.name_scope('input'):
            self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps),
                                         name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps),
                                          name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            if not self.use_embedding:
                self.rnn_inputs = tf.one_hot(self.inputs, self.n_classes)
            else:
                embedding = tf.get_variable('embedding', [self.n_classes, self.embedding_size])
                self.rnn_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def _build_net(self):
        """
        Multiple layer multi-step lstm network
        :return: 
        """

        def create_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('LSTM'):
            # Multi-layer lstm
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [create_cell(self.lstm_size, self.keep_prob) for _ in range(self.n_layers)])
            self.init_state = cell.zero_state(self.batch_size,
                                              tf.float32)  # state is the hidden state

            # Multi-step (for a sequence data)
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.rnn_inputs,
                                                                    initial_state=self.init_state)

            # The lstm output is multi-step output
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])  # row: bath_size * num_steps

            with tf.variable_scope('output'):
                w = tf.Variable(tf.truncated_normal([self.lstm_size, self.n_classes], stddev=0.1))
                b = tf.Variable(tf.zeros(self.n_classes))
                self.logits = tf.matmul(x, w) + b
                self.preds = tf.nn.softmax(self.logits, name='prob_pred')

    def _build_loss(self):
        with tf.name_scope('loss'):
            label = tf.one_hot(self.targets, self.n_classes)
            y = tf.reshape(label, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y)
            self.loss = tf.reduce_mean(loss)

    def _build_optimizer(self):
        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_vars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimizer = optimizer.apply_gradients(zip(grads, train_vars))

    def train(self, batch_generator, max_steps, save_path, save_interval, log_interval):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_state = sess.run(self.init_state)
            for x, y in batch_generator:
                start = time.time()
                feed_dict = {self.inputs: x, self.targets: y, self.keep_prob: self.train_keep_prob,
                             self.init_state: new_state}
                batch_loss, new_state, prob, _ = sess.run(
                    [self.loss, self.final_state, self.preds, self.optimizer], feed_dict=feed_dict)
                step += 1
                end = time.time()

                # print out
                if step % log_interval == 0:
                    print('Step: {}, Loss: {:.4f}, Time: {:.4f}'.format(step, batch_loss,
                                                                        end - start))
                if step % save_interval == 0:
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

                if step > max_steps:
                    return

    def predict(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        new_state = self.session.run(self.init_state)
        preds = np.ones((vocab_size,))
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = c

            feed_dict = {self.inputs: x,
                         self.keep_prob: 1.,
                         self.init_state: new_state}
            preds, new_state = self.session.run([self.preds, self.final_state], feed_dict=feed_dict)

        c = pick_top_n(preds, vocab_size)
        samples.append(c)

        for _ in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.init_state: new_state}
            preds, new_state = self.session.run([self.preds, self.final_state],
                                             feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load_model(self, ckpt_path):
        self.session = tf.Session()
        try:
            self.saver.restore(self.session, ckpt_path)
        except:
            print('Cannot restore model from ', ckpt_path)
