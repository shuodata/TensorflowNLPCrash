#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
# print sys.getdefaultencoding()

import collections
import datetime
import logging
import numpy as np
import os
import pprint
from scipy import ndimage
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

# Define hyperparameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log")
flags.DEFINE_string("train_tfrecords_file",
                    "./data/cancer/cancer_train.csv.tfrecords",
                    "The glob pattern of train TFRecords files")
flags.DEFINE_string("validate_tfrecords_file",
                    "./data/cancer/cancer_test.csv.tfrecords",
                    "The glob pattern of validate TFRecords files")
flags.DEFINE_integer("feature_size", 9, "Number of feature size")
flags.DEFINE_integer("label_size", 2, "Number of label size")
flags.DEFINE_float("learning_rate", 0.01, "The learning rate")
flags.DEFINE_integer("epoch_number", 1000, "Number of epochs to train")
flags.DEFINE_integer("batch_size", 1024, "The batch size of training")
flags.DEFINE_integer("validate_batch_size", 1024,
                     "The batch size of validation")
flags.DEFINE_integer("batch_thread_number", 1,
                     "Number of threads to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "The minimal number after dequeue")
flags.DEFINE_string("checkpoint_path", "./checkpoint/",
                    "The path of checkpoint")
flags.DEFINE_string("output_path", "./tensorboard/",
                    "The path of tensorboard event files")
flags.DEFINE_string("model", "dnn", "Support dnn, lr, wide_and_deep")
flags.DEFINE_string("model_network", "128 32 8", "The neural network of model")
flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization or not")
flags.DEFINE_float("bn_epsilon", 0.001, "The epsilon of batch normalization")
flags.DEFINE_boolean("enable_dropout", False, "Enable dropout or not")
flags.DEFINE_float("dropout_keep_prob", 0.5, "The dropout keep prob")
flags.DEFINE_boolean("enable_lr_decay", False, "Enable learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.96, "Learning rate decay rate")
flags.DEFINE_string("optimizer", "adagrad", "The optimizer to train")
flags.DEFINE_integer("steps_to_validate", 1000,
                     "Steps to validate and print state")
flags.DEFINE_string("mode", "train", "Support train, export, inference")
flags.DEFINE_string("model_path", "./model/", "The path of the model")
flags.DEFINE_integer("model_version", 1, "The version of the model")
flags.DEFINE_string("inference_test_file", "./data/cancer_test.csv",
                    "The test file for inference")
flags.DEFINE_string("inference_result_file", "./inference_result.txt",
                    "The result file from inference")


def main():
  # Get hyperparameters
  if FLAGS.enable_colored_log:
    import coloredlogs
    coloredlogs.install()
  logging.basicConfig(level=logging.INFO)
  FEATURE_SIZE = FLAGS.feature_size
  LABEL_SIZE = FLAGS.label_size
  EPOCH_NUMBER = FLAGS.epoch_number
  if EPOCH_NUMBER <= 0:
    EPOCH_NUMBER = None
  BATCH_THREAD_NUMBER = FLAGS.batch_thread_number
  MIN_AFTER_DEQUEUE = FLAGS.min_after_dequeue
  BATCH_CAPACITY = BATCH_THREAD_NUMBER * FLAGS.batch_size + MIN_AFTER_DEQUEUE
  MODE = FLAGS.mode
  MODEL = FLAGS.model
  CHECKPOINT_PATH = FLAGS.checkpoint_path
  if not CHECKPOINT_PATH.startswith("fds://") and not os.path.exists(
      CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
  CHECKPOINT_FILE = CHECKPOINT_PATH + "/checkpoint.ckpt"
  LATEST_CHECKPOINT = tf.train.latest_checkpoint(CHECKPOINT_PATH)
  OUTPUT_PATH = FLAGS.output_path
  if not OUTPUT_PATH.startswith("fds://") and not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
  pprint.PrettyPrinter().pprint(FLAGS.__flags)

  # Read data file for all poetries
  poetries = []
  POETRY_FILE = "./data/chinese_poetry/poetry.txt"

  with open(POETRY_FILE, "r") as f:  
    for line in f:
      try:
        title, content = line.decode("UTF-8").strip().split(":")
        # Remove the useless content
        if "_" in content or "-" in content or "(" in content or ")" in content or "（" in content or "）" in content or "<" in content or ">" in content or "《" in content or "》" in content or "[" in content or "]" in content or ' ' in content:
          continue
        content = content.replace(",", "").replace(".", "").replace("，", "").replace("。", "")
        poetries.append(content)
      except Exception as e:
        logging.error("Get exception: {}".format(e))  

  poetry_number = len(poetries)
  logging.info("Poetry number: {}".format(poetry_number))

  # Encode the chinese characters with numbers
  # The map<string, int> with 85 items
  char_id_map = {}
  id_char_map = {}
  index = 0
  for poetry in poetries:
    for char in poetry:
      if char not in char_id_map:
        char_id_map[char] = index
        id_char_map[index] = char
        index += 1

  vocab_number = len(char_id_map)
  logging.info("Chinese char number: {}".format(vocab_number))

  # Convert chinese characters into numbers
  # The array of [int] with 42500 items
  encoded_poetries = []
  for poetry in poetries:
    encoded_poetry = []
    for char in poetry:
      encoded_poetry.append(char_id_map[char])
    encoded_poetries.append(encoded_poetry)

  assert(len(encoded_poetries) == poetry_number)

  # Generate features and labels
  train_features = []
  train_labels = []

  for encoded_poetry in encoded_poetries:
    # For example: [0, 63, 20, 18, 16, 56]
    one_features = []
    # For example: [63, 20, 18, 16, 56, 56]
    one_labels = []

    for i in range(len(encoded_poetry)):
      one_features.append(encoded_poetry[i])
      # Handle the edge case and use the last one as input and output
      if i >= len(encoded_poetry) - 1:
        one_labels.append(encoded_poetry[i])
      else:
        one_labels.append(encoded_poetry[i + 1])

    train_features.append(one_features)
    train_labels.append(one_labels)

  # Define the model
  logging.info("Use the model: {}, model network: {}".format(
      MODEL, FLAGS.model_network))
  '''
  if MODEL == 'rnn':
    cell_fun = tf.nn.rnn_cell.BasicRNNCell
  elif MODEL == 'gru':
    cell_fun = tf.nn.rnn_cell.GRUCell
  elif MODEL == 'lstm':
    cell_fun = tf.nn.rnn_cell.BasicLSTMCell
  '''
  # TODO: support batch size for training, which causes ValueError: 'setting an array element with a sequence.'
  BATCH_SIZE = 1
  input_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, None])
  output_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, None])
  
  rnn_hidden_number = 128
  rnn_layer_number = 2

  cell = tf.contrib.rnn.BasicLSTMCell(rnn_hidden_number, state_is_tuple=True)
  cell = tf.contrib.rnn.MultiRNNCell([cell] * rnn_layer_number, state_is_tuple=True)
 
  initial_state = cell.zero_state(BATCH_SIZE, tf.float32)

  # [128, 85]
  softmax_w = tf.get_variable("softmax_w", [rnn_hidden_number, vocab_number])
  # [85]
  softmax_b = tf.get_variable("softmax_b", [vocab_number])
  # [85, 128]
  embedding = tf.get_variable("embedding", [vocab_number, rnn_hidden_number])
  # [1, None, 128]
  inputs = tf.nn.embedding_lookup(embedding, input_placeholder)
  # [1, None, 128], last_state is [1, 128]
  outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
  
  # [None, 128]
  output = tf.reshape(outputs, [-1, rnn_hidden_number])
 
  # [None, 85]
  logits = tf.matmul(output, softmax_w) + softmax_b

  targets = tf.reshape(output_placeholder, [-1])
  loss = tf.reduce_mean(tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], vocab_number))
  global_step = tf.Variable(0, name="global_step", trainable=False)
  learning_rate = FLAGS.learning_rate
  optimizer = get_optimizer(FLAGS.optimizer, learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)

  # For infernce
  # [BATCH_SIZE, FEATURE_SIZE]
  inference_softmax = tf.nn.softmax(logits)
  # TODO: may only work when batch size is 1
  # [FEATURE_SIZE]
  inference_softmax_max = tf.argmax(inference_softmax, 1)

  # Initialize saver and summary
  saver = tf.train.Saver()
  tf.summary.scalar("loss", loss)
  # TODO: not work for the latest TensorFlow
  #summary_op = tf.summary.merge_all()
  
  # Create session to run
  with tf.Session() as sess:
    logging.info("Start to run with mode: {}".format(MODE))
    #writer = tf.summary.FileWriter(OUTPUT_PATH, sess.graph)
    sess.run(tf.global_variables_initializer())

    if MODE == "train":
      restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
      start_time = datetime.datetime.now()
      
      batch_number = poetry_number // BATCH_SIZE
      for epoch in range(100):
        for i in range(batch_number):
          x = train_features[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
          y = train_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
          if len(x) > 0 and len(x[0]) > 0:
            try:
              #_, loss_value = sess.run([train_op, loss], feed_dict={input_placeholder: [train_features[i]], output_placeholder: [train_labels[i]]})
              _, loss_value = sess.run([train_op, loss], feed_dict={input_placeholder: x, output_placeholder: y})
            except Exception as e:
              print(e)
              exit(1)


          if i % FLAGS.steps_to_validate == 0:
            #_, loss_value, step, summary_value  = sess.run([train_op, loss, global_step, summary_op], feed_dict={input_placeholder: [train_features[i]], output_placeholder: [train_labels[i]]})
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={input_placeholder: [train_features[i]], output_placeholder: [train_labels[i]]})
            end_time = datetime.datetime.now()
            logging.info("[{}] Step: {}, loss: {}".format(end_time - start_time, step, loss_value))
            saver.save(sess, CHECKPOINT_FILE, global_step=step)
            start_time = end_time
            #writer.add_summary(summary_value, step)

    elif MODE == "inference":
      if not restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT):
        logging.error("No checkpoint found, exit now")
        exit(1)

      start_time = datetime.datetime.now()

      last_state_value = sess.run(cell.zero_state(1, tf.float32))
      # [[1]]
      x = [[train_features[1][0]]]
      x = [[18]]
      softmax_value, softmax_max_value, last_state_value = sess.run([inference_softmax, inference_softmax_max, last_state], feed_dict={input_placeholder: x, initial_state: last_state_value})

      generated_poetry_length = 39
      generated_poetry_chars = [id_char_map[x[0][0]]]
      for i in range(generated_poetry_length):
        print(softmax_max_value[0])
        x = [[softmax_max_value[0]]]
        generated_char = id_char_map[softmax_max_value[0]]
        generated_poetry_chars.append(generated_char)

        softmax_value, softmax_max_value, last_state_value = sess.run([inference_softmax, inference_softmax_max, last_state], feed_dict={input_placeholder: x, initial_state: last_state_value})
       
      end_time = datetime.datetime.now()
      logging.info("[{}] Inference result: {}".format(end_time - start_time, generated_poetry_chars))

      generated_poetry = ""
      for i in range(len(generated_poetry_chars)):
        generated_poetry += generated_poetry_chars[i]
        if (i + 1) % 8 == 0:
          generated_poetry += "。\n"
        elif (i + 1) % 4 == 0:
          generated_poetry += "，"
      print(generated_poetry)



  exit(0)
  


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
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_units)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * stacked_layer_nubmer)
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
  tf.summary.scalar('loss', loss)
  init_op = tf.global_variables_initializer()

  # Create session to run graph
  with tf.Session() as sess:
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
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

def get_optimizer(optimizer, learning_rate):
  logging.info("Use the optimizer: {}".format(optimizer))
  if optimizer == "sgd":
    return tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer == "adadelta":
    return tf.train.AdadeltaOptimizer(learning_rate)
  elif optimizer == "adagrad":
    return tf.train.AdagradOptimizer(learning_rate)
  elif optimizer == "adam":
    return tf.train.AdamOptimizer(learning_rate)
  elif optimizer == "ftrl":
    return tf.train.FtrlOptimizer(learning_rate)
  elif optimizer == "rmsprop":
    return tf.train.RMSPropOptimizer(learning_rate)
  else:
    logging.error("Unknow optimizer, exit now")
    exit(1)

def restore_session_from_checkpoint(sess, saver, checkpoint):
  if checkpoint:
    logging.info("Restore session from checkpoint: {}".format(checkpoint))
    saver.restore(sess, checkpoint)
    return True
  else:
    return False

if __name__ == "__main__":
  main()
