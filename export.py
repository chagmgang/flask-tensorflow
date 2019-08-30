import modeling

import tensorflow as tf
import numpy as np

mnist_classifier = tf.estimator.Estimator(
    model_fn=modeling.cnn_model_fn, model_dir='tmp/mnist_convnet_model')

def serving_input_fn():
    receiver_tensors = {
        'image': tf.placeholder(tf.float32, [None, 784], name='image')}
    features = {
        'x': receiver_tensors['image']}
    return tf.estimator.export.ServingInputReceiver(
        receiver_tensors=receiver_tensors,
        features=features)

mnist_classifier.export_savedmodel('export/mnist_convnet_model', serving_input_fn)