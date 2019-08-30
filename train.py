import modeling

import tensorflow as tf
import numpy as np

((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=1000)

mnist_classifier = tf.estimator.Estimator(
    model_fn=modeling.cnn_model_fn, model_dir='tmp/mnist_convnet_model',
    config=run_config)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

mnist_classifier.train(
    input_fn=train_input_fn)