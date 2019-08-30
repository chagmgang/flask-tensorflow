import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1], name='input_layer')
    print(input_layer)
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='conv1')
    print(conv1)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='conv2')
    print(conv2)
    flat = tf.layers.flatten(inputs=conv2, name='flat')
    print(flat)
    dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu, name='dense1')
    print(dense1)
    predict = tf.layers.dense(inputs=dense1, units=10, activation=tf.nn.softmax, name='predict')
    print(predict)

    predictions = {
        'classes': tf.argmax(input=predict, axis=1),
        'probabilities': predict}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    answer = tf.one_hot(labels, 10)
    loss = tf.reduce_mean(-answer * tf.log(predict + 1e-8))
    accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)