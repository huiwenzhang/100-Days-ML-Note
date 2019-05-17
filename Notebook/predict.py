"""
Genearte a sequence based on the trained rnn model
"""
import tensorflow as tf
from utilis import TextConverter
from rnn_model import CharRNN

import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', '', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')


def main(argv):
    FLAGS.start_string = FLAGS.start_string.encode('utf-8').decode('utf-8')
    converter = TextConverter(vocab_file=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size,
                    sampling=True,
                    lstm_size=FLAGS.lstm_size,
                    n_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.load_model(FLAGS.checkpoint_path)

    start = converter.text2arr(FLAGS.start_string)
    arr = model.predict(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr2text(arr))


if __name__ == "__main__":
    tf.app.run()
