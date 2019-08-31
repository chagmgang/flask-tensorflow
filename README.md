# Process from training with estimator to serving with flask

* Dataset: MNIST dataset from *Keras*
* Training Method : Tensorflow Estimator

## File explaination

* modeling.py \rightarrow make model for tensorflow estimator
* train.py \rightarrow tensorflow training process
* eval.py \rightarrow evaluate tensorflow estimator model based on train.py
* export.py \rightarrow export freeze model from model made by train.py
* serving_flask.py \rightarrow serving freeze model with flask method

## Requirements

* tensorflow==1.14.0
* numpy
* flask
* flask-restful
