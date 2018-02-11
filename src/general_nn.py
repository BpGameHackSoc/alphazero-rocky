import abc
from keras.layers import Activation, Conv2D, Dense, Flatten, Input
from keras.models import Sequential, load_model, Model
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform
from src.config import DEFAULT_NEURAL_NET_SETTINGS

class NeuralNetwork(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model_path=None):      
        self.config()

    def config(self):
        self.config = {}
        for key, value in DEFAULT_NEURAL_NET_SETTINGS.items():
            self.config[key] = value

    def init_model(self, model_path):
        if model_path is None:
            self.new_model(self.config['input_shape'])
        else:
            self.load(model_path)

    def new_model(self, input_shape):
        inp = Input(input_shape)
        x = self.conv_layer(inp, 'firstconv_')
        for i in range(self.config['res_layer_n']):
            x = self.res_layer(x, i+1)
        value = self.value_head(x)
        policy = self.policy_head(x)
        model = Model(inp, [value, policy])
        model.compile(loss=['mean_squared_error', 'categorical_crossentropy'],
                      optimizer='rmsprop')
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
        x = self.conv_layer(x, 'value_', filter_n=1, kernel_size=1)
        x = Flatten()(x)
        x = Activation('relu')(x)
        x = Dense(self.config['value_hidden_size'])(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('tanh')(x)
        return x

    def policy_head(self, x):
        x = self.conv_layer(x, 'policy_', filter_n=2, kernel_size=1)
        x = Flatten()(x)
        x = Dense(self.config['no_of_possible_actions'])(x)
        x = Activation('sigmoid')(x)
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
            data_format='channels_first',
            name=prefix+"_conv"+suffix
        )(x)
        x = BatchNormalization(axis=1, name=prefix+"_batchnorm"+suffix)(x)
        if not original_x is None:
            x = Add(name=prefix+"_sum"+suffix)([original_x, x])
        x = Activation("relu", name=prefix+"_relu"+suffix)(x)
        return x

    def learn(self, states, values, probabilities, save=False, relative_path=None):
        states = np.array(states)
        values = np.array(values)
        probabilities = np.array(probabilities)
        history = self.model.fit(
            states,
            [values, probabilities],
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            verbose=self.config['verbose']
        )
        self.history.extend(history)
        if save:
            self.save(relative_path)

    def predict(self, states):
        return self.model.predict(states)

    def save(self, relative_path=None):
        relative_path = self.__random_string(15) if relative_path is None else relative_path
        path = 'bin/' + relative_path + '.h5'
        self.model.save(path)
        print('Model saved in ' + path)

    def load(self, path):
        self.model = load_model(path)

    def __random_string(self, n):
        return ''.join(np.random.choice(list(string.ascii_lowercase+ string.digits), n))