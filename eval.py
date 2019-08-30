import modeling

import tensorflow as tf
import numpy as np

((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

mnist_classifier = tf.estimator.Estimator(
    model_fn=modeling.cnn_model_fn, model_dir='tmp/mnist_convnet_model')

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)