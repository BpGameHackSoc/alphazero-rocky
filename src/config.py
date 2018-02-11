
# ==============================================================
# NEURAL NET
# ==============================================================

DEFAULT_NEURAL_NET_SETTINGS = {
    'no_of_possible_actions' : None,      # The number of actions the softmax layer produces
    'value_hidden_size' : 64,             # Size of hidden layer in value head 
    'res_layer_n' : 3,                    # The number of residual layers
    'filter_n' : 128,                     # The number of filters in a conv layer
    'kernel_size' : 3,
    'batch_size' : 20,                    
    'epochs' : 5,
    'verbose' : 0,
    'input_shape' : None,
    'history' : []
}
