#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import seq2seq

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', 10, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 32,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("optimizer", "adam", "optimizer to train")
flags.DEFINE_integer('steps_to_validate', 1,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train", "Opetion mode: train, inference")
flags.DEFINE_string("image", "./data/inference/Pikachu.png",
                    "The image to inference")
flags.DEFINE_string("inference_start_word", "l", "The start word to inference")
flags.DEFINE_string(
    "model", "stacked_lstm",
    "Model to train, option model: lstm, bidirectional_lstm, stacked_lstm")


def main():
  print("Start generating lycrics")

  # Initialize train and test data
  batch_size = FLAGS.batch_size
  epoch_number = FLAGS.epoch_number
  sequece_length = 20
  rnn_hidden_units = 100
  stacked_layer_nubmer = 3

  # TODO: Use python 3 for encoding for Chinese
  #lycrics_filepath = "./data/jay_lyrics.txt"
  lycrics_filepath = "./data/shakespeare.txt"
  #with open(lycrics_filepath) as f:
  import codecs
  f = codecs.open(lycrics_filepath, encoding='utf-8')
  lycrics_data = f.read()

  words = list(set(lycrics_data))
  words.sort()
  vocabulary_size = len(words)
  char_id_map = {}
  id_char_map = {}
  for index, char in enumerate(words):
    id_char_map[index] = char
    char_id_map[char] = index

  train_dataset = []
  train_labels = []
  index = 0
  for i in range(batch_size):
    features = lycrics_data[index:index + sequece_length]
    labels = lycrics_data[index + 1:index + sequece_length + 1]
    index += sequece_length

    features = [char_id_map[word] for word in features]
    labels = [char_id_map[word] for word in labels]

    train_dataset.append(features)
    train_labels.append(labels)

  # Define the model
  batch_size = FLAGS.batch_size
  mode = FLAGS.mode

  if mode == "inference":
    batch_size = 1
    sequece_length = 1

  x = tf.placeholder(tf.int32, shape=(None, sequece_length))
  y = tf.placeholder(tf.int32, shape=(None, sequece_length))
  epoch_number = FLAGS.epoch_number
  checkpoint_dir = FLAGS.checkpoint_dir
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  tensorboard_dir = FLAGS.tensorboard_dir

  checkpoint_file = checkpoint_dir + "/checkpoint.ckpt"
  steps_to_validate = FLAGS.steps_to_validate

  def lstm_inference(x):
    pass

  def stacked_lstm_inference(x):
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_hidden_units)
    lstm_cells = rnn_cell.MultiRNNCell([lstm_cell] * stacked_layer_nubmer)
    initial_state = lstm_cells.zero_state(batch_size, tf.float32)

    with tf.variable_scope("stacked_lstm"):
      weights = tf.get_variable("weights", [rnn_hidden_units, vocabulary_size])
      bias = tf.get_variable("bias", [vocabulary_size])
      embedding = tf.get_variable("embedding", [vocabulary_size,
                                                rnn_hidden_units])

    inputs = tf.nn.embedding_lookup(embedding, x)
    outputs, last_state = tf.nn.dynamic_rnn(lstm_cells,
                                            inputs,
                                            initial_state=initial_state)

    output = tf.reshape(outputs, [-1, rnn_hidden_units])
    logits = tf.add(tf.matmul(output, weights), bias)

    return logits, lstm_cells, initial_state, last_state

  def inference(inputs):
    print("Use the model: {}".format(FLAGS.model))
    if FLAGS.model == "lstm":
      return lstm_inference(inputs)
    elif FLAGS.model == "stacked_lstm":
      return stacked_lstm_inference(inputs)
    else:
      print("Unknow model, exit now")
      exit(1)

  # Define train op
  logits, lstm_cells, initial_state, last_state = inference(x)
  #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit,
  #                                                                     y))

  targets = tf.reshape(y, [-1])
  loss = seq2seq.sequence_loss_by_example(
      [logits],
      [targets],
      [tf.ones_like(targets,
                    dtype=tf.float32)])
  loss = tf.reduce_sum(loss)

  predict_softmax = tf.nn.softmax(logits)

  learning_rate = FLAGS.learning_rate
  print("Use the optimizer: {}".format(FLAGS.optimizer))
  if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif FLAGS.optimizer == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif FLAGS.optimizer == "ftrl":
    optimizer = tf.train.FtrlOptimizer(learning_rate)
  elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    print("Unknow optimizer: {}, exit now".format(FLAGS.optimizer))
    exit(1)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)

  saver = tf.train.Saver()
  tf.scalar_summary('loss', loss)
  init_op = tf.initialize_all_variables()

  # Create session to run graph
  with tf.Session() as sess:
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
    sess.run(init_op)

    if mode == "train":
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Continue training from the model {}".format(
            ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

      start_time = datetime.datetime.now()
      for epoch in range(epoch_number):

        _, loss_value, step = sess.run(
            [train_op, loss, global_step],
            feed_dict={x: train_dataset,
                       y: train_labels})

        if epoch % steps_to_validate == 0:
          end_time = datetime.datetime.now()

          print("[{}] Epoch: {}, loss: {}".format(end_time - start_time, epoch,
                                                  loss_value))

          saver.save(sess, checkpoint_file, global_step=step)
          #writer.add_summary(summary_value, step)
          start_time = end_time

    elif mode == "inference":
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("Load the model {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

      start_time = datetime.datetime.now()

      word = FLAGS.inference_start_word
      generate_word_number = 100
      generate_lyrics = word

      state = sess.run(lstm_cells.zero_state(1, tf.float32))

      for i in range(generate_word_number):
        x2 = np.zeros((1, 1))
        x2[0, 0] = char_id_map[word]

        prediction, state = sess.run(
            [predict_softmax, last_state],
            feed_dict={x: x2,
                       initial_state: state})
        predict_word_id = np.argmax(prediction[0])

        word = id_char_map[predict_word_id]
        generate_lyrics += word

      end_time = datetime.datetime.now()
      print("[{}] Generated lyrics:\n{}".format(end_time - start_time,
                                                generate_lyrics))

    else:
      print("Unknow mode, please choose 'train' or 'inference'")

  print("End of generating lycrics")


if __name__ == "__main__":
  main()
