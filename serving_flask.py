import flask
import flask_restful

import tensorflow as tf
import numpy as np

from tensorflow.python.saved_model import tag_constants

## read data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

## load exported model
export_dir = 'export/mnist_convnet_model/1567154280'

graph = tf.get_default_graph()
sess = tf.Session(graph=graph)

tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)

## get_tensor_by_name
# show all name of tensor
# list_of_tuples = [op.values() for op in graph.get_operations()]
tensor_output_ids = graph.get_tensor_by_name('predict/Softmax:0')
tensor_input_ids = graph.get_tensor_by_name('image:0')

## flask server
app = flask.Flask(__name__)
api = flask_restful.Api(app)

class Test(flask_restful.Resource):
    def get(self):

        random_index = np.random.choice(eval_data.shape[0])
        test_data = eval_data[random_index]
        test_data = np.reshape(test_data, [784])
        test_answer = eval_labels[random_index]
        result = sess.run(tensor_output_ids, feed_dict={
            tensor_input_ids: [test_data]})

        return {
            # 'data': test_data.tolist(),
            'predict': np.argmax(result, axis=1).tolist(),
            'answer': test_answer.tolist()
        }

api.add_resource(Test, '/api/test')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)