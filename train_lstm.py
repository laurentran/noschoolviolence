import tensorflow as tf
from tensorflow import data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random
import shutil
import multiprocessing
import itertools

# set parameters and load data
TARGET_LABELS = ['Weapons', 'Bullying', 'Physical', 'Dating']
MODEL_NAME = 'model'
TRAIN_DATA_SIZE = 297 
NUM_EPOCHS = 100
BATCH_SIZE = 20
TOTAL_STEPS = (TRAIN_DATA_SIZE/BATCH_SIZE)*NUM_EPOCHS

hparams  = tf.contrib.training.HParams(
    num_epochs = NUM_EPOCHS,
    batch_size = BATCH_SIZE,
    forget_bias=1.0,
    learning_rate=0.01,
    keep_prob = 0.8,
    max_steps = TOTAL_STEPS
)

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.estimator.RunConfig(
    tf_random_seed=321,
    model_dir=model_dir
)

features_path = 'features.npy'
labels_path = 'labels.npy'

X = np.load(features_path)
Y = np.load(labels_path)

# randomly shuffle data
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
Y = Y[s]

# split data into train and test sets
train_X = X[0:296,:,:]
test_X = X[297:369,:,:]

train_Y = Y[0:296,:]
test_Y = Y[297:369,:]


def create_features_labels(features, labels):
    return (features, labels)

def rnn_model_fn(features, labels, mode, params):
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(
        num_units=100, 
        forget_bias=params.forget_bias,
        activation=tf.nn.relu)]

    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    outputs, _ = tf.nn.static_rnn(cell=multi_rnn_cell, 
                                inputs=features, 
                                dtype=tf.float32)
    
    outputs = outputs[-1]
    logits = tf.layers.dense(inputs=outputs,
                                  units=len(TARGET_LABELS),
                                  activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.sigmoid(logits)
        predicted_indices = tf.argmax(probabilities, 1)
        predictions = {
            'class': tf.gather(TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # Calculate loss using sigmoid cross entropy
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels))
    
    tf.summary.scalar('loss', loss)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss, 
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.sigmoid(logits)
        predicted_indices = tf.argmax(probabilities, 1)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(mode, 
                                          loss=loss, 
                                          eval_metric_ops=eval_metric_ops)


def create_estimator(run_config, hparams):
    
    estimator = tf.estimator.Estimator(model_fn=rnn_model_fn, 
                                  params=hparams, 
                                  config=run_config)
    return estimator


train_spec = tf.estimator.TrainSpec(
    input_fn = lambda: create_features_labels(train_X, train_Y),
    max_steps=hparams.max_steps,
    hooks=None
)

eval_spec = tf.estimator.EvalSpec(
    input_fn = lambda: create_features_labels(test_X, test_Y),
    steps=None
)

estimator = create_estimator(run_config, hparams)

tf.estimator.train_and_evaluate(
    estimator=estimator,
    train_spec=train_spec, 
    eval_spec=eval_spec
)