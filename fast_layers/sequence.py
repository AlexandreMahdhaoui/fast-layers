import tensorflow as tf


class Sequence(tf.keras.layers.Layer):
    """
    Arguments:
        name: str, positional arg
        inputs: str: name of input pipe/connector | list: names of input pipes/connectors, positional arg
        sequence=None: list of keras.layers objects,
        is_output_layer=False,
        trainable=True,
    Attributes:
        inputs: str or list of input names.
        sequence: list of keras.layers objects,
        is_output_layer: True if this is the output Sequence of a Layer object.
    Methods:
        call(x, training=False): by calling the sequence through __call__(), computes x.
        self_build(): build the layers of the sequence into this Sequence object.
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
            assert(type(sequence) == list), 'Please provide a sequence of one or more keras.layers objects as a list'
        self.sequence = sequence

    def call(self, x, training=False):
        for layer in self.sequence:
            x = layer(x, training=training)
        return x

    def self_build(self):
        for layer in self.sequence:
            self.__dict__[layer.name] = layer
