from __future__ import print_function

import sys
import threading

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import time
import utils

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_integer('batch_size', 1, 'Number of tests per query')
tf.app.flags.DEFINE_integer('concurrency', 1, 'Maximum number of concurrent batched inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test values')
tf.app.flags.DEFINE_string('server', 'insteadthread.corp.gq1.yahoo.com:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('test_data', '../data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx.1k', 'Path to test data')
tf.app.flags.DEFINE_integer('total_features', 156408, 'Number of features in the test data set')
FLAGS = tf.app.flags.FLAGS

path = '../data_yahoo/dataset2/featindex_25m_thres10.txt'
INPUT_DIM, FIELD_OFFSETS = utils.initiate(path)
total_features = FLAGS.total_features # specific to data set
#Global results array to store results retrieved by _callback
num_batches = FLAGS.num_tests/FLAGS.batch_size
if (num_batches * FLAGS.batch_size) != FLAGS.num_tests:
    num_batches += 1
results = [None] * num_batches


class _ResultCounter(object):
    """Coordinator, tracks active batch requests and throttles as necessary"""

    def __init__(self, concurrency):
        self._concurrency = concurrency
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_complete(self):
        with self._condition:
            while self._done != num_batches:
                self._condition.wait()

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

def _create_rpc_callback(batch_number, result_counter):
    """Creates RPC callback function.
    Args:
      label: The correct label for the predicted example.
      result_counter: Counter for the prediction result.
    Returns:
      The callback function.
    """
    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
          result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            results[batch_number] = np.array(result_future.result().outputs['output'].float_val)
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback

def line2one_hot_batch(line_batch, total_features, batch_size):
    """Convert train input from dens to sparse arrays

    Args:
      line_batch: list of raw data from test set in sparse features "labels | features" format
      total_features: number of features in test data set
      batch_size: number of tests to convert and include
    Returns:
      Array [batch_size] of labels
      Array [batch_size, total_features], a dense array of features for each test in the batch
    """
    if batch_size > len(line_batch):
        batch_size = len(line_batch)
    label = np.zeros((batch_size,1))
    one_hot_array = np.zeros((batch_size, total_features), dtype=np.float32)
    for batch_index in range(batch_size):
        line = line_batch[batch_index].strip()
        if line[0] == '1':
            label[batch_index,0] = 1
        else:
            label[batch_index,0] = 0
        begin_pos = line.find('|')
        feature_array = line[ begin_pos + 2 : ].split(' ')
        for feature_index in feature_array:
            one_hot_array[batch_index, int(feature_index)-1 ] = 1
    return label, one_hot_array

def do_inference(hostport, test_data, concurrency, num_tests, batch_size):
    """Tests PredictionService with concurrent-batched requests.

    Args:
      hostport: Host:port address of the PredictionService.
      test_data: The full path to the test data set.
      concurrency: Maximum number of concurrent requests.
      num_tests: Number of test tensors to use.
      batch_size: Number of tests to include in each query

    Returns:
      The results of the queries

    Raises:
      IOError: An error occurred processing test data set.
    """
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_counter = _ResultCounter(concurrency)
    num_field = 15

    with open(test_data, 'r') as f:
        data = f.read().split('\n')
        requests = []
        t0 = time.time()

        #Create batches of requests
        for i in range(num_batches):
            data_batch = data[:batch_size]
            data = data[batch_size:]

            index_list = [[] for _ in range(num_field)]
            values_list = [[] for _ in range(num_field)]
            for j in range(batch_size):
                X, y = utils.slice(utils.process_lines([data_batch[j]], 'fwfm', INPUT_DIM, FIELD_OFFSETS), 0, -1)
                for idx in range(num_field):
                    index_list[idx].append(X[idx][0].tolist())
                    values_list[idx].append(1)

            requests.append(predict_pb2.PredictRequest())
            requests[i].model_spec.name = 'serve'
            requests[i].model_spec.signature_name = 'model'
            requests[i].output_filter.append('outputs')
            for idx in range(num_field):
                requests[i].inputs["field_" + str(idx) + "_values"].CopyFrom(tf.contrib.util.make_tensor_proto(values_list[idx], shape=[len(values_list[idx])], dtype=tf.int64))
                requests[i].inputs["field_" + str(idx) + "_indices"].CopyFrom(tf.contrib.util.make_tensor_proto(index_list[idx], shape=[len(index_list[idx]), 2], dtype=tf.float32))
                requests[i].inputs["field_" + str(idx) + "_dense_shape"].CopyFrom(tf.contrib.util.make_tensor_proto([batch_size, total_features], shape=[2], dtype=tf.int64))
        t1 = time.time()

        #Query server
        for i in range(num_batches):
            result_counter.throttle()
            result = stub.Predict.future(requests[i], 100.0)  # 100 secs timeout
            result.add_done_callback(_create_rpc_callback(i, result_counter))
        t2 = time.time()

        #Synchronize on comleted queries
        result_counter.get_complete()
        t3 = time.time()
        full_results = []
        for values in results:
            full_results.extend(values)

        print("Elapsed time for ", num_tests, " request creations: ", (t1-t0))
        print("Elapsed time for ", num_batches, " batch submissions: ", (t2-t1))
        print("Elapsed time for ", num_tests, " inferences: ", (t3-t1))
        return full_results

def main(_):
    if not FLAGS.server:
        print('please specify server host:port')
        return
    do_inference(FLAGS.server, FLAGS.test_data, FLAGS.concurrency, FLAGS.num_tests, FLAGS.batch_size)


if __name__ == '__main__':
    tf.app.run()
