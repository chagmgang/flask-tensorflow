# Process from training with estimator to serving with flask

* Dataset: MNIST dataset from *Keras*
* Training Method : Tensorflow Estimator

## File explaination

* modeling.py : make model for tensorflow estimator
* train.py : tensorflow training process
* eval.py : evaluate tensorflow estimator model based on train.py
* export.py : export freeze model from model made by train.py
* serving_flask.py : serving freeze model with flask method

## Requirements

* tensorflow==1.14.0
* numpy
* flask
* flask-restful

## Command line

```
python train.py ## training
python eval.py ## evaluate
python export.py ## export model
python serving_flask.py ## serving by flask, you have to edit line 17(export file path)
```

## Serving and Check by requests module

### Serving
```
(xxx) xxx@xxx:~/flask-tensorflow$ python serving_flask.py 
WARNING: Logging before flag parsing goes to stderr.
W0831 03:19:47.895947 139689095231296 deprecation_wrapper.py:119] From serving_flask.py:19: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

W0831 03:19:47.904812 139689095231296 deprecation_wrapper.py:119] From serving_flask.py:20: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
...
...
XLA_FLAGS=--xla_hlo_profile.
 * Serving Flask app "serving_flask" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
I0831 03:19:48.059808 139689095231296 _internal.py:122]  * Running on http://0.0.0.0:3000/ (Press CTRL+C to quit)
I0831 03:20:00.875686 139682033678080 _internal.py:122] 127.0.0.1 - - [31/Aug/2019 03:20:00] "GET /api/test HTTP/1.1" 200 -
```

### Check
```
Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import requests
>>> res = requests.get('http://0.0.0.0:3000/api/test')
>>> print(res.json())
{'predict': [3], 'answer': 3}
```
