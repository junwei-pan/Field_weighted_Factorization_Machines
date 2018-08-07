import cPickle as pkl
import tensorflow as tf
import utils
import numpy as np
from math import factorial

def comb(n, k):
    return int( factorial(n) / factorial(k) / factorial(n - k) )

dtype = utils.DTYPE

gpu_device = '/gpu:0'

class LR:
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_weight=0, random_seed=None):
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w']
            b = self.vars['b']
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = xw + b
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)) + \
                        l2_weight * tf.nn.l2_loss(xw)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {self.X: X}
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class FM:
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('v', [input_dim, factor_order], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']

            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v))
            p = 0.5 * tf.reshape(
                tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(v)), 1),
                [-1, output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = xw + b + p
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {self.X: X}
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class FNN:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, init_path=None,
                 opt_algo='gd', learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            for i in range(1, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class CCPM:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, init_path=None, opt_algo='gd',
                 learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        embedding_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = embedding_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('f1', [embedding_order, layer_sizes[2], 1, 2], 'tnormal', dtype))
        init_vars.append(('f2', [embedding_order, layer_sizes[3], 2, 2], 'tnormal', dtype))
        init_vars.append(('w1', [2 * 3 * embedding_order, 1], 'tnormal', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            l = tf.nn.dropout(
                utils.activate(
                    tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i]
                               for i in range(num_inputs)], 1),
                    layer_acts[0]),
                layer_keeps[0])
            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embedding_order, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    num_inputs / 2),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                utils.activate(
                    tf.reshape(l, [-1, embedding_order * 3 * 2]),
                    layer_acts[1]),
                layer_keeps[1])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[2]),
                layer_keeps[2])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class PNN1:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            # This is where W_p \cdot p happens.
            # k1 is \theta, which is the weight for each field(feature) vector
            p = tf.reduce_sum(
                tf.reshape(
                    tf.matmul(
                        tf.reshape(
                            tf.transpose(
                                tf.reshape(l, [-1, num_inputs, factor_order]),
                                [0, 2, 1]),
                            [-1, num_inputs]),
                        k1),
                    [-1, factor_order, layer_sizes[2]]),
                1)
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1 + p,
                    layer_acts[1]),
                layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class Fast_CTR:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        init_vars.append(('bf', [factor_order], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            bf = self.vars['bf']
            l = utils.activate(
                    tf.reshape(
                        tf.reduce_sum(
                            tf.reshape(l, [-1, num_inputs, factor_order]), 1
                        ), [-1, factor_order]
                    ) + bf , 'none'
            )

            w1 = self.vars['w1']
            b1 = self.vars['b1']

            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[1]),
                layer_keeps[1])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                pass
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class Fast_CTR_Concat:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            l = tf.reshape(
                tf.concat(
                    tf.reshape(l, [-1, num_inputs, factor_order]),
                    1
                ), [-1, num_inputs * factor_order]
            )

            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']

            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[1]),
                layer_keeps[1])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class PNN1_Fixed:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None, has_field_bias=True):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w_l', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w_p', [num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        #init_vars.append(('w1', [num_inputs * factor_order + num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            if has_field_bias:
                x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            else:
                x = tf.concat([xw[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])

            w_l = self.vars['w_l']
            w_p = self.vars['w_p']
            b1 = self.vars['b1']
            # This is where W_p \cdot p happens.
            # k1 is \theta, which is the weight for each field(feature) vector
            p = tf.matmul(
                tf.reshape(l, [-1, num_inputs, factor_order]),
                tf.transpose(
                    tf.reshape(l, [-1, num_inputs, factor_order]), [0, 2, 1])
            )

            p = tf.nn.dropout(
                utils.activate(
                    tf.matmul(
                        tf.reshape(p, [-1, num_inputs * num_inputs]),
                        w_p),
                    'none'
                ),
                layer_keeps[1]
            )

            l = utils.activate(tf.matmul(l, w_l) + b1 + p,
                    layer_acts[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    if i == 1:
                        self.loss += layer_l2[i] * tf.nn.l2_loss(w_l)
                        self.loss += layer_l2[i] * tf.nn.l2_loss(w_p)
                    else:
                        wi = self.vars['w%d' % i]
                        # bi = self.vars['b%d' % i]
                        self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                pass
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class FwFM3:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None, has_field_bias=False, \
                 l2_dict=None, fullLayer3 = True, survivors = 15):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0]) # Number of fields, i.e., M
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i] # Number of unique features for field i
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w_l', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w_p', [num_inputs * (num_inputs-1)/2, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w_3', [num_inputs * (num_inputs-1)/2, num_inputs], 'tnormal', dtype))
        #init_vars.append(('w1', [num_inputs * factor_order + num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(gpu_device):
                if random_seed is not None:
                    tf.set_random_seed(random_seed)
                self.X = [tf.sparse_placeholder(dtype, name='x' + str(i)) for i in range(num_inputs)]
                self.y = tf.placeholder(dtype)
                self.vars = utils.init_var_map(init_vars, init_path)
                w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
                b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            # xw = [x / tf.norm(x) for x in xw]

            with tf.device(gpu_device):
                if has_field_bias:
                    x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
                else:
                    x = tf.concat([xw[i] for i in range(num_inputs)], 1)
                l = tf.nn.dropout(
                    utils.activate(x, layer_acts[0]),
                    layer_keeps[0])

                w_l = self.vars['w_l']
                w_p = self.vars['w_p']
                w_3 = self.vars['w_3']
                b1 = self.vars['b1']
                # This is where W_p \cdot p happens.
                # k1 is \theta, which is the weight for each field(feature) vector

            index_left = []
            index_right = []

            for i in range(num_inputs):
                for j in range(i+1,num_inputs):
                    index_left.append(i)
                    index_right.append(j)

            with tf.device(gpu_device):
                l_ = tf.reshape(l, [-1, num_inputs, factor_order])
                l_left = tf.gather(l_, index_left, axis = 1)
                l_right = tf.gather(l_, index_right, axis =1)
                p_full = tf.multiply(l_left, l_right)
                p = tf.reduce_sum(p_full, 2)
                p = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(p, w_p),
                        'none'
                    ),
                    layer_keeps[1]
                )

                ## Now for the Interaction Layer 3
                l_trans = tf.transpose(l_, [ 0, 2, 1])
                p3_full = tf.matmul(p_full, l_trans)
                p3_full = tf.multiply(p3_full, w_3)
                p3 = tf.reduce_sum(p3_full, 2)

                if fullLayer3:
                    p3 = tf.reduce_sum(p3, 1, keep_dims = True)
                
                else:
                    [_, top_indexes] = tf.nn.top_k(
                        tf.abs(p3),
                        k=survivors,
                        sorted=True,
                        name=None
                    )
                    p3 = tf.reduce_sum(tf.gather(p3, top_indexes, axis = 1), 1, keep_dims = True)

                l = utils.activate(
                        tf.matmul(l, w_l) + b1 + p + p3,
                        layer_acts[1])

                for i in range(2, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    bi = self.vars['b%d' % i]
                    l = tf.nn.dropout(
                        utils.activate(
                            tf.matmul(l, wi) + bi,
                            layer_acts[i]),
                        layer_keeps[i])

                self.y_prob = tf.sigmoid(l, name='y')
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
                # l2 regularization for the linear weights, embeddings and r
                if l2_dict is not None:
                    if l2_dict.has_key('linear_w'):
                        self.loss += l2_dict['linear_w'] * tf.nn.l2_loss(w_l)
                    if l2_dict.has_key('r'):
                        self.loss += l2_dict['r'] * tf.nn.l2_loss(w_p)
                    if l2_dict.has_key('v'):
                        for i in range(num_inputs):
                            self.loss += l2_dict['v'] * tf.nn.l2_loss(self.vars['w0_%d' % i])

                self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            config.log_device_placement=False
            self.sess = tf.Session(config=config)
            with tf.device(gpu_device):
                tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

 
class FwFM:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None, has_field_bias=False, l2_dict=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0]) # Number of fields, i.e., M
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i] # Number of unique features for field i
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w_l', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w_p', [num_inputs * (num_inputs-1)/2, layer_sizes[2]], 'tnormal', dtype))
        #init_vars.append(('w1', [num_inputs * factor_order + num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(gpu_device):
                if random_seed is not None:
                    tf.set_random_seed(random_seed)
                self.X = [tf.sparse_placeholder(dtype, name='x' + str(i)) for i in range(num_inputs)]
                self.y = tf.placeholder(dtype)
                self.vars = utils.init_var_map(init_vars, init_path)
                w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
                b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            # xw = [x / tf.norm(x) for x in xw]

            with tf.device(gpu_device):
                if has_field_bias:
                    x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
                else:
                    x = tf.concat([xw[i] for i in range(num_inputs)], 1)
                l = tf.nn.dropout(
                    utils.activate(x, layer_acts[0]),
                    layer_keeps[0])

                w_l = self.vars['w_l']
                w_p = self.vars['w_p']
                b1 = self.vars['b1']
                # This is where W_p \cdot p happens.
                # k1 is \theta, which is the weight for each field(feature) vector

            index_left = []
            index_right = []

            for i in range(num_inputs):
                for j in range(i + 1, num_inputs):
                    index_left.append(i)
                    index_right.append(j)

            with tf.device(gpu_device):
                l_ = tf.reshape(l, [-1, num_inputs, factor_order])
                l_left = tf.gather(l_, index_left, axis = 1)
                l_right = tf.gather(l_, index_right, axis = 1)
                p_full = tf.multiply(l_left, l_right)
                p = tf.reduce_sum(p_full, 2)

                p = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(p, w_p),
                        'none'
                    ),
                    layer_keeps[1]
                )

                l = utils.activate(tf.matmul(l, w_l) + b1 + p, 
                    layer_acts[1])

                for i in range(2, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    bi = self.vars['b%d' % i]
                    l = tf.nn.dropout(
                        utils.activate(
                            tf.matmul(l, wi) + bi,
                            layer_acts[i]),
                        layer_keeps[i])

                self.y_prob = tf.sigmoid(l, name='y')
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
                # l2 regularization for the linear weights, embeddings and r
                if l2_dict is not None:
                    if l2_dict.has_key('linear_w'):
                        self.loss += l2_dict['linear_w'] * tf.nn.l2_loss(w_l)
                    if l2_dict.has_key('r'):
                        self.loss += l2_dict['r'] * tf.nn.l2_loss(w_p)
                    if l2_dict.has_key('v'):
                        for i in range(num_inputs):
                            self.loss += l2_dict['v'] * tf.nn.l2_loss(self.vars['w0_%d' % i])

                self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            config.log_device_placement=False
            self.sess = tf.Session(config=config)
            with tf.device(gpu_device):
                tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class MultiTask_FwFM:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None, int_path=None,
                 opt_algo='gd', learning_rate=1e-2, random_seed=None, has_field_bias=False, l2_dict=None,
                 num_lines=None, index_lines=None, flag_r_factorized=False):
        """
        :param layer_sizes: [num_fields, factor_layer, l_p size]
        :param layer_acts:
        :param layer_keeps:
        :param int_path:
        :param opt_algo:
        :param learning_rate:
        :param random_seed:
        :param has_field_bias:
        :param l2_dict:
        """

        print 'num_lines', num_lines
        init_vars = []
        num_inputs = len(layer_sizes[0])-1 # Here we minus one to exclude conv_type as a feature
        factor_order = layer_sizes[1]
        # Parameters initialization
        for i in range(num_inputs+1):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
        for i in range(num_inputs):
            init_vars.append(('b0_%d' % i, [factor_order], 'zero', dtype)) # ? Why layer_output?
        init_vars.append(('w_l', [num_lines, num_inputs * factor_order], 'tnormal', dtype))
        init_vars.append(('r', [num_lines, num_inputs*(num_inputs-1)/2], 'tnormal', dtype))
        init_vars.append(('r_factorized', [num_lines, num_inputs, factor_order], 'tnormal', dtype))
        init_vars.append(('b1', [num_lines, layer_sizes[2]], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            config.gpu_options.allow_growth = True
            config.log_device_placement = False
            self.sess = tf.Session(config=config)
            with tf.device(gpu_device):
                if random_seed is not None:
                    tf.set_random_seed(random_seed)
                self.X = [tf.sparse_placeholder(dtype, name='x%d' % i) for i in range(num_inputs+1)]
                self.y = tf.placeholder(dtype)
                self.vars = utils.init_var_map(init_vars, int_path)
                w0 = [self.vars['w0_%d' % i] for i in range(num_inputs+1)]
                # The dimension of b0 is only num_inputs since it's used only after lookup
                b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs+1) if i != index_lines]
            with tf.device(gpu_device):
                if has_field_bias:
                    x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
                else:
                    x = tf.concat(xw, 1)
                    #x = tf.concat([xw[i] for i in range(num_inputs)], 1)
                print 'x.shape', x.shape
                l = tf.nn.dropout(
                    utils.activate(x, layer_acts[0]),
                    layer_keeps[0]
                )

                #w_l = self.vars['w_l']
                #b1 = self.vars['b1']
                #w_p = self.vars['r_1' ]

            index_left = []
            index_right = []
            for i in range(num_inputs):
                for j in range(i + 1, num_inputs):
                    index_left.append(i)
                    index_right.append(j)
            if flag_r_factorized:
                r_factorized = self.vars['r_factorized']
                r_product = []
                for i in range(num_lines):
                    r_factorized_i = tf.convert_to_tensor(r_factorized[i,:])
                    r_left = tf.gather(r_factorized_i, index_left)
                    r_right = tf.gather(r_factorized_i, index_right)
                    r_p = tf.multiply(r_left, r_right)
                    r_p = tf.reduce_sum(r_p, 1)
                    r_product.append(r_p)
                r = tf.sparse_tensor_dense_matmul(self.X[index_lines], r_product)
                #r_factorized = tf.sparse_tensor_dense_matmul(self.X[index_lines], self.vars['r_factorized'])
            else:
                r = tf.sparse_tensor_dense_matmul(self.X[index_lines], self.vars['r'])

            w_l = tf.sparse_tensor_dense_matmul(self.X[index_lines], self.vars['w_l'])
            b1 = tf.sparse_tensor_dense_matmul(self.X[index_lines], self.vars['b1'])
            w_l = tf.reshape(w_l, [-1, num_inputs * factor_order])
            b1 = tf.reshape(b1, [-1, layer_sizes[2]])

            with tf.device(gpu_device):
                l_ = tf.reshape(l, [-1, num_inputs, factor_order])
                l_left = tf.gather(l_, index_left, axis = 1)
                l_right = tf.gather(l_, index_right, axis = 1)
                p = tf.multiply(l_left, l_right)
                p = tf.reduce_sum(p, 2)
                p = tf.nn.dropout(
                    utils.activate(
                        tf.reduce_sum(
                            tf.multiply(p, r), 
                            1, keep_dims = True),
                        'none'),
                    layer_keeps[1]
                )

                l = utils.activate(tf.reduce_sum(tf.multiply(l, w_l), 1, keep_dims = True)
                    + b1
                    + p,
                    layer_acts[1])

                for i in range(2, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    bi = self.vars['b%d' % i]
                    l = tf.nn.dropout(
                        utils.activate(
                            tf.matmul(l, wi) + bi,
                            layer_acts[i]),
                        layer_keeps[i])

                self.y_prob = tf.sigmoid(l, name='y')
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y)
                )

                if l2_dict is not None:
                    if l2_dict.has_key('linear_w'):
                        self.loss += l2_dict['linear_w'] * tf.nn.l2_loss(w_l)
                    if l2_dict.has_key('r'):
                        self.loss += l2_dict['r'] * tf.nn.l2_loss(self.vars['r'])
                    if l2_dict.has_key('v'):
                        for i in range(num_inputs):
                            self.loss += l2_dict['v'] * tf.nn.l2_loss(self.vars['w0_%d' % i])
                self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)
            with tf.device(gpu_device):
                tf.global_variables_initializer().run(session=self.sess)


    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at ', model_path
        
class DINN:
    def __init__(self, layer_sizes=None, allLayer2=True, layer_acts=None, layer_keeps=None, layer_l2=None, \
                 kernel_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None, \
                 has_field_bias=False, l2_dict=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """

        print'layer_sizes', layer_sizes
        num_fields = len(layer_sizes[0])  # Number of fields, i.e., M
        print'num_fields', str(num_fields)
        embd_card = layer_sizes[1]
        print'embd_card', str(embd_card)
        print 'layer_sizes', layer_sizes
        num_layers = layer_sizes[3]
        print'num_layers', str(num_layers)

        init_vars = []
        num_inputs = len(layer_sizes[0])  # Number of fields, i.e., M
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))

        for i in range(3, num_layers):
            init_vars.append(('w_pooling_%d' % i, [layer_sizes[5][i - 1] * num_fields, embd_card], 'tnormal', dtype))

        init_vars.append(('w_l', [num_inputs * embd_card, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w2', [comb(num_fields, 2), embd_card], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():

            indexes_Left_Deep = [[], [], []]
            indexes_Right_Deep = [[], [], []]
            for i in range(3, num_layers):
                indexes_Left_Deep.append(
                    tf.constant(np.repeat(np.arange(layer_sizes[5][i - 1]), num_fields), dtype=tf.int32))
                indexes_Right_Deep.append(
                    tf.constant(np.tile(np.arange(num_fields), layer_sizes[5][i - 1]), dtype=tf.int32))

            with tf.device(gpu_device):
                if random_seed is not None:
                    tf.set_random_seed(random_seed)
                self.X = [tf.sparse_placeholder(dtype, name='x' + str(i)) for i in range(num_inputs)]
                self.y = tf.placeholder(dtype)
                self.vars = utils.init_var_map(init_vars, init_path)
                w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
                b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
                w_pooling = [self.vars['w_pooling_%d' % i] for i in range(3, num_layers)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            # xw = [x / tf.norm(x) for x in xw]

            with tf.device(gpu_device):
                if has_field_bias:
                    x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
                else:
                    x = tf.concat([xw[i] for i in range(num_inputs)], 1)
                l = tf.nn.dropout(
                    utils.activate(x, layer_acts[0]),
                    layer_keeps[0])

                w_l = self.vars['w_l']
                w2 = self.vars['w2']
                # w_p = self.vars['w_p']
                b1 = self.vars['b1']
                # This is where W_p \cdot p happens.
                # k1 is \theta, which is the weight for each field(feature) vector

            index_left = []
            index_right = []

            for i in range(num_inputs):
                for j in range(num_inputs - i - 1):
                    index_left.append(i)
                    index_right.append(i + j + 1)

            with tf.device(gpu_device):
                dirImport = 0
                l_trans = tf.reshape(l, [-1, num_inputs, factor_order])
                combToBeUsedInNext = [0, 0]
                for i in range(2, num_layers):
                    combToBeUsedInNext.append(0)
                #print "l_trans.get_shape()"
                #print l_trans.get_shape()
                for i in range(2, num_layers):
                    current_layer = i
                    #print 'current_layer: ', str(current_layer)
                    if current_layer == 2:
                        l_left = tf.gather(l_trans, index_left, axis=1)
                        #print "l_left.get_shape()"
                        #print l_left.get_shape()

                        l_right = tf.gather(l_trans, index_right, axis=1)
                        #print "l_right.get_shape()"
                        #print l_right.get_shape()

                        p2 = tf.multiply(l_left, l_right)
                        #print "p2.get_shape()"
                        #print p2.get_shape()

                        p2W = tf.multiply(p2, w2)
                        #print "p2W.get_shape()"
                        #print p2W.get_shape()

                        p2W_lessDim2 = tf.reduce_sum(p2W, 2)
                        #print "p2W_lessDim2.get_shape()"
                        #print p2W_lessDim2.get_shape()
                        p2W_oneDimAbs = tf.reduce_sum(tf.abs(p2W_lessDim2), 0)
                        #print "p2W_oneDimAbs.get_shape()"
                        #print p2W_oneDimAbs.get_shape()

                        [_, topCurrIndexes] = tf.nn.top_k(
                            p2W_oneDimAbs,
                            k=layer_sizes[4][current_layer],
                            sorted=True,
                            name=None
                        )

                        if allLayer2:
                            dirImport += tf.reduce_sum(p2W_lessDim2, axis=1, keep_dims=True)
                        else:
                            dirImport += tf.reduce_sum(tf.gather(p2W_lessDim2, topCurrIndexes, axis=1), axis=1,
                                                       keep_dims=True)

                        combToBeUsedInNext[2] = tf.gather(p2, topCurrIndexes[0:layer_sizes[5][current_layer]], axis=1)
                        #print "combToBeUsedInNext.get_shape()"
                        #print combToBeUsedInNext[2].get_shape()
                    else:
                        #print "indexes_Left_Deep[current_layer].get_shape()"
                        #print indexes_Left_Deep[current_layer].get_shape()

                        x_left = tf.gather(combToBeUsedInNext[current_layer - 1], indexes_Left_Deep[current_layer],
                                           axis=1)
                        #print "x_left.get_shape()"
                        #print x_left.get_shape()

                        x_right = tf.gather(l_trans, indexes_Right_Deep[current_layer], axis=1)
                        #print "x_right.get_shape()"
                        #print x_right.get_shape()

                        pComb = tf.multiply(x_left, x_right)
                        #print "pComb.get_shape()"
                        #print pComb.get_shape()

                        pCombW = tf.multiply(pComb, w_pooling[current_layer - 3])
                        #print "pCombW.get_shape()"
                        #print pCombW.get_shape()

                        pCombW_lessDim2 = tf.reduce_sum(pCombW, 2)
                        #print "p2W_lessDim2.get_shape()"
                        #print pCombW_lessDim2.get_shape()

                        pCombWAbs_lessDim2 = tf.reduce_sum(tf.abs(pCombW_lessDim2), 0)
                        #print "pCombWAbs_lessDim2.get_shape()"
                        #print pCombWAbs_lessDim2.get_shape()

                        [_, topCurrIndexes] = tf.nn.top_k(
                            pCombWAbs_lessDim2,
                            k=layer_sizes[4][current_layer],
                            sorted=True,
                            name=None
                        )

                        dirImport += tf.reduce_sum(tf.gather(pCombW_lessDim2, topCurrIndexes, axis=1), axis=1,
                                                   keep_dims=True)
                        if current_layer < num_layers - 1:
                            combToBeUsedInNext[current_layer] = tf.gather(pComb, topCurrIndexes[
                                                                                 0:layer_sizes[5][current_layer]],
                                                                          axis=1)
                            #print "combToBeUsedInNext.get_shape()"
                            #print combToBeUsedInNext[current_layer].get_shape()

                l = utils.activate(
                    tf.matmul(l, w_l) + b1 + dirImport,
                    layer_acts[1])

                self.y_prob = tf.sigmoid(l, name='y')

                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))

                self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            config.log_device_placement = False
            self.sess = tf.Session(config=config)
            with tf.device(gpu_device):
                tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class FwFM_LE:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None, has_field_bias=False, l2_dict=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
        """
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            # w0 store the embeddings for all features.
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            #init_vars.append(('w_l', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
            init_vars.append(('w_l_v_all_%d' % i, [layer_input, 1], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        #init_vars.append(('w_l', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('w_p', [num_inputs * (num_inputs-1)/2, layer_sizes[2]], 'tnormal', dtype))
        #init_vars.append(('w1', [num_inputs * factor_order + num_inputs * num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(gpu_device):
                if random_seed is not None:
                    tf.set_random_seed(random_seed)
                self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
                self.y = tf.placeholder(dtype)
                self.vars = utils.init_var_map(init_vars, init_path)
                w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
                w_l_v_all = [self.vars['w_l_v_all_%d' % i] for i in range(num_inputs)]
                b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            # Multiply SparseTensor X[i] by dense matrix w0[i]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            w_l_v = [tf.sparse_tensor_dense_matmul(self.X[i], w_l_v_all[i]) for i in range(num_inputs)]
            #w_l_v = tf.reshape(w_l_v, [num_inputs * factor_order, layer_sizes[2]])
            with tf.device(gpu_device):
                w_l_v = tf.reshape(w_l_v, [-1, num_inputs])
                if has_field_bias:
                    x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
                else:
                    x = tf.concat([xw[i] for i in range(num_inputs)], 1)
                l = tf.nn.dropout(
                    utils.activate(x, layer_acts[0]),
                    layer_keeps[0])

                #w_l = self.vars['w_l']
                w_p = self.vars['w_p']
                b1 = self.vars['b1']
                # This is where W_p \cdot p happens.
                # k1 is \theta, which is the weight for each field(feature) vector

            index_left = []
            index_right = []

            for i in range(num_inputs):
                for j in range(num_inputs - i - 1):
                    index_left.append(i)
                    index_right.append(i + j + 1)

            with tf.device(gpu_device):
                l_trans = tf.transpose(tf.reshape(l, [-1, num_inputs, factor_order]), [1, 0, 2])
                l_left = tf.gather(l_trans, index_left)
                l_right = tf.gather(l_trans, index_right)
                p = tf.transpose(tf.multiply(l_left, l_right), [1, 0, 2])
                p = tf.reduce_sum(p, 2)
                p = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(
                            tf.reshape(p, [-1, num_inputs*(num_inputs-1)/2]),
                            w_p),
                        'none'
                    ),
                    layer_keeps[1]
                )

                l = utils.activate(
                        tf.reduce_sum(w_l_v, 1, keep_dims=True) + b1 + p,
                        #tf.matmul(l, w_l) + b1 + p,
                        layer_acts[1])

                for i in range(2, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    bi = self.vars['b%d' % i]
                    l = tf.nn.dropout(
                        utils.activate(
                            tf.matmul(l, wi) + bi,
                            layer_acts[i]),
                        layer_keeps[i])

                self.y_prob = tf.sigmoid(l)

                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
                # l2 regularization for the linear weights, embeddings and r
                if l2_dict is not None:
                    if l2_dict.has_key('linear_w'):
                        self.loss += l2_dict['linear_w'] * tf.nn.l2_loss(w_l_v)
                    if l2_dict.has_key('r'):
                        self.loss += l2_dict['r'] * tf.nn.l2_loss(w_p)
                    if l2_dict.has_key('v'):
                        self.loss += l2_dict['v'] * tf.nn.l2_loss(w_l_v)
                        for i in range(num_inputs):
                            self.loss += l2_dict['v'] * tf.nn.l2_loss(self.vars['w0_%d' % i])

                self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            config.log_device_placement=False
            self.sess = tf.Session(config=config)
            with tf.device(gpu_device):
                tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class FFM:

    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None, init_path=None, opt_algo='gd', learning_rate=0.01, random_seed=None, has_field_bias=False, l1_dict=None, l2_dict=None):
        """
        # Arguments:
            layer_size: [num_fields, factor_layer, l_p size]
            layer_acts: ["tanh", "none"]
            layer_keep: [1, 1]
            layer_l2: [0, 0]
            kernel_l2: 0
            has_field_bias: False
            l1_dict: {}
            le_dict: {}
        """
        init_vars = []
        num_inputs = len(layer_sizes[0]) # Number of fields, i.e., M.
        factor_order = layer_sizes[1] # Dimension of embedding vectors, i.e., k.
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i] # Number of unique features for field i.
            layer_output = factor_order
            init_vars.append(('w_l_v_all_%d' % i, [layer_input, 1], 'tnormal', dtype))
            init_vars.append(('w0_%d' % i, [layer_input, num_inputs * layer_output], 'tnormal', dtype))

        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device(gpu_device):
                if random_seed is not None:
                    tf.set_random_seed(random_seed)
                self.X = [ tf.sparse_placeholder(dtype) for i in range(num_inputs) ]
                self.y = tf.placeholder(dtype)
                self.vars = utils.init_var_map(init_vars, init_path)
                w0 = [ self.vars['w0_%d' % i] for i in range(num_inputs) ]
                w_l_v_all = [ self.vars['w_l_v_all_%d' % i] for i in range(num_inputs) ]
            xw = [ tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs) ]
            xw1 = [ tf.sparse_tensor_dense_matmul(self.X[i], w_l_v_all[i]) for i in range(num_inputs) ]
            with tf.device(gpu_device):
                xw1 = tf.reshape(xw1, [-1, num_inputs])
                x = tf.concat([ xw[i] for i in range(num_inputs) ], 1)
                l = tf.nn.dropout(utils.activate(x, layer_acts[0]), layer_keeps[0])
                b1 = self.vars['b1']

            index_left = []
            index_right = []
            for i in range(num_inputs):
                for j in range(num_inputs):
                    if i != j:
                        index_left.append(i * num_inputs + j)
                        index_right.append(j * num_inputs + i)

            with tf.device(gpu_device):
                l_trans = tf.transpose(tf.reshape(l, [-1, num_inputs * num_inputs, factor_order]), [1, 0, 2])
                l_left = tf.gather(l_trans, index_left)
                l_right = tf.gather(l_trans, index_right)
                p = tf.transpose(tf.multiply(l_left, l_right), [1, 0, 2])
                p = tf.reduce_sum(p, 2)
                p = tf.nn.dropout(utils.activate(tf.reshape(p, [-1, num_inputs * (num_inputs - 1)]), 'none'), layer_keeps[1])
                l = utils.activate(tf.reduce_sum(xw1, 1, keep_dims=True) + b1, layer_acts[1])
                l = utils.activate(l + tf.reduce_sum(p, 1, keep_dims=True), layer_acts[1])
                self.y_prob = tf.sigmoid(l)
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
                if l2_dict is not None:
                    if l2_dict.has_key('linear_w'):
                        self.loss += l2_dict['linear_w'] * tf.nn.l2_loss(w_l_v)
                    if l2_dict.has_key('v'):
                        for i in range(num_inputs):
                            self.loss += l2_dict['v'] * tf.nn.l2_loss(self.vars['w0_%d' % i])

                if l1_dict is not None:
                    if l1_dict.has_key('linear_w'):
                        l1_regularizer = tf.contrib.layers.l1_regularizer(
                            scale=l1_dict['linear_w'], scope=None
                        )
                        penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [xw1])
                        self.loss += penalty
                    if l1_dict.has_key('r'):
                        l1_regularizer = tf.contrib.layers.l1_regularizer(
                            scale=l1_dict['r'], scope=None
                        )
                        penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [w_p])
                        #penalty = utils.l1_loss(w_p, l1_dict['r'])
                        self.loss += penalty
                    if l1_dict.has_key('v'):
                        l1_regularizer = tf.contrib.layers.l1_regularizer(
                            scale=l1_dict['v'], scope=None
                        )
                        for i in range(num_inputs):
                            penalty += tf.contrib.layers.apply_regularization(l1_regularizer, [self.vars['w0_%d' % i]])
                        self.loss += penalty

                self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            config.log_device_placement = False
            self.sess = tf.Session(config=config)
            with tf.device(gpu_device):
                tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]

        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)

        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path

class PNN2:
    def __init__(self, layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, kernel_l2=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [factor_order * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                layer_keeps[0])
            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            z = tf.reduce_sum(tf.reshape(l, [-1, num_inputs, factor_order]), 1)
            p = tf.reshape(
                tf.matmul(tf.reshape(z, [-1, factor_order, 1]),
                          tf.reshape(z, [-1, 1, factor_order])),
                [-1, factor_order * factor_order])
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + tf.matmul(p, k1) + b1,
                    layer_acts[1]),
                layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    layer_keeps[i])

            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

    def run(self, fetches, X=None, y=None):
        feed_dict = {}
        for i in range(len(X)):
            feed_dict[self.X[i]] = X[i]
        if y is not None:
            feed_dict[self.y] = y
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path
