import tensorflow as tf


class Sequence(tf.keras.layers.Layer):
    """
    Arguments:
        name: str,
        sequence: list of layers objects,
        inputs: str: name of input pipe/connector | list: names of input pipes/connectors
        is_output_layer=False,
        trainable=True,
    Attributes:
        _
    Methods:
        _
    """

    def __init__(self,
                 name: str,
                 inputs: str or list,
                 sequence=None,
                 is_output_layer=False,
                 trainable=True,
                 **kwargs):
        super(Sequence, self).__init__(name=name, trainable=trainable, **kwargs)
        if type(inputs) is not str:
            assert(type(inputs) == list and all(type(_) is str for _ in inputs)), \
                'Please either provide a single input as a string, or several inputs as a list of string'
        self.inputs = inputs
        self.is_output_layer = is_output_layer
        if sequence is not None:
            assert(type(sequence) == list), 'Please provide a sequence of one or more layers as a list'
        self.sequence = sequence

    def __call__(self, x, training=False):
        for layer in self.sequence:
            x = layer(x, training=training)
        return x

    def self_build(self):
        for layer in self.sequence:
            self.__dict__[layer.name] = layer
