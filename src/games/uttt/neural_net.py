import abc
from keras.layers import Activation, Conv2D, Dense, Flatten, Input,Concatenate
from keras.models import Sequential, load_model, Model
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform
from keras.regularizers import l2
from keras import optimizers
import keras
from src.config import DEFAULT_NEURAL_NET_SETTINGS, WORK_FOLDER
import numpy as np
import string
from src.general_nn import NeuralNetwork

l2_reg = 0.001

class UTTTNN(NeuralNetwork):
    def __init__(self, model_name=None):
        self.config()
        self.init_model(model_name)

    def config(self):
        self.config = {}
        for key, value in DEFAULT_NEURAL_NET_SETTINGS.items():
            self.config[key] = value

    def init_model(self, model_name):
        if model_name is None:
            self.new_model()
        else:
            self.load(model_name)

    def new_model(self):
        inp = Input((9,9,1))
        x = Conv2D(
            filters=16,
            kernel_size=(3,3),
            padding='valid',
            strides=(3,3),
            kernel_regularizer=l2(l2_reg),
            data_format='channels_last',
            name="conv"
        )(inp)
        x = Flatten()(x)
        hand_features = Input((1054,))
        x = keras.layers.concatenate([x,hand_features])
        x = Dense(128,activation='relu',kernel_regularizer=l2(l2_reg))(x)
        value = Dense(16,activation='relu',kernel_regularizer=l2(l2_reg))(x)
        value = Dense(1, activation='tanh', kernel_regularizer=l2(l2_reg/10))(value)
        policy = Dense(100,activation='relu',kernel_regularizer=l2(l2_reg))(x)
        policy = Dense(81, activation='softmax', kernel_regularizer=l2(l2_reg))(policy)
        model = Model([inp,hand_features], [value, policy])
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=['mean_squared_error', 'categorical_crossentropy'],
                      optimizer=sgd)
        self.model = model

    def res_layer(self, x, index):
        '''
            Creates a single residual layer.
        '''
        original_x = x
        prefix = 'res_' + str(index)
        x = self.conv_layer(x, prefix, '_1')
        x = self.conv_layer(x, prefix, '_2', original_x)
        return x

    def value_head(self, x):
        x = self.conv_layer(x, 'value_', filter_n=1, kernel_size=1, kernel_regularizer=l2(0.001))
        x = Flatten()(x)
        x = Activation('linear')(x)
        x = Dense(self.config['value_hidden_size'], kernel_regularizer=l2(0.001))(x)
        x = Activation('relu')(x)
        x = Dense(1, kernel_regularizer=l2(0.01))(x)
        x = Activation('tanh', name='value')(x)
        return x

    def policy_head(self, x):
        x = self.conv_layer(x, 'policy_', filter_n=2, kernel_size=1, kernel_regularizer=l2(0.01))
        x = Flatten()(x)
        x = Dense(self.config['no_of_possible_actions'], kernel_regularizer=l2(0.01))(x)
        x = Activation('softmax', name='policy')(x)
        return x

    def conv_layer(self, x, prefix, suffix='', original_x=None, **kwargs):
        '''
            Creates a single convolutional layer, often as:
                [convolution] --> [batch_norm] --> [activation]
            Params:
                x :             the input layer
                prefix :        the prefix of all layers' names
                suffix :        the suffix of all layers' names
                original_x:     a previous layer that wants to be concatenated to this
                                layer. It is used as half of a residual block:
                                [convolution] --> [batch_norm] --> [concat] --> [activation]
                                If None, it leaves the concatenation.
        '''
        filter_n = kwargs.get('filter_n', self.config['filter_n'])
        kernel_size = kwargs.get('kernel_size', self.config['kernel_size'])
        x = Conv2D(
            filters=filter_n,
            kernel_size=kernel_size,
            padding='same',
            strides=1,
            kernel_regularizer=l2(0.01),
            data_format='channels_first',
            name=prefix+"_conv"+suffix
        )(x)
        x = BatchNormalization(axis=1, name=prefix+"_batchnorm"+suffix)(x)
        if not original_x is None:
            x = Add(name=prefix+"_sum"+suffix)([original_x, x])
        x = Activation("relu", name=prefix+"_relu"+suffix)(x)
        return x

    def learn(self, states, values, probabilities):
        vanilla_hand = list(zip(*states))
        vanilla_board = np.array(vanilla_hand[0])
        hand_features = np.array(vanilla_hand[1])
        values = np.array(values)
        probabilities = np.array(probabilities)
        history = self.model.fit(
            [vanilla_board,hand_features],
            [values, probabilities],
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            verbose=self.config['verbose'],
            validation_split=self.config['validation_split'],
            shuffle=True
        )

    def predict(self, states):
        return self.model.predict(states)

    def save(self, file_name=None, to_print=True):
        file_name = self.__random_string(15) if file_name is None else file_name
        path = WORK_FOLDER + file_name + '.h5'
        self.model.save(path)
        if to_print:
            print('Model saved in ' + path)

    def load(self, file_name):
        self.model = load_model(WORK_FOLDER + file_name + '.h5')

    def __random_string(self, n):
        return ''.join(np.random.choice(list(string.ascii_lowercase+ string.digits), n))






















### FOR REFERENCE

# batch_size = 16
# patch_size = 5
# depth = 16
# num_hidden = 64
# num_steps = 1001
# stride = 2
#
# graph = tf.Graph()
#
# with graph.as_default():
#     # Input data.
#     tf_train_dataset = tf.placeholder(
#         tf.float32, shape=(batch_size, image_size, image_size, num_channels))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables.
#     layer1_weights = tf.Variable(tf.truncated_normal(
#         [patch_size, patch_size, num_channels, depth], stddev=0.1))
#     layer1_biases = tf.Variable(tf.zeros([depth]))
#     layer2_weights = tf.Variable(tf.truncated_normal(
#         [patch_size, patch_size, depth, depth], stddev=0.1))
#     layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
#     layer3_weights = tf.Variable(tf.truncated_normal(
#         [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
#     layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
#     layer4_weights = tf.Variable(tf.truncated_normal(
#         [num_hidden, num_labels], stddev=0.1))
#     layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
#
#
#     # Model.
#     def model(data):
#         conv = tf.nn.conv2d(data, layer1_weights, [1, stride, stride, 1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer1_biases)
#         conv = tf.nn.conv2d(hidden, layer2_weights, [1, stride, stride, 1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer2_biases)
#         shape = hidden.get_shape().as_list()
#         reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#         hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#         return tf.matmul(hidden, layer4_weights) + layer4_biases
#
#
#     # Training computation.
#     logits = model(tf_train_dataset)
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
#
#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#
#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
#     test_prediction = tf.nn.softmax(model(tf_test_dataset))
#
# with tf.Session(graph=graph) as session:
#     tf.global_variables_initializer().run()
#     print('Initialized')
#     for step in range(num_steps):
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#         batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#         feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
#         _, l, predictions = session.run(
#             [optimizer, loss, train_prediction], feed_dict=feed_dict)
#         if (step % 50 == 0):
#             print('Minibatch loss at step %d: %f' % (step, l))
#             print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#             print('Validation accuracy: %.1f%%' % accuracy(
#                 valid_prediction.eval(), valid_labels))
#     print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))