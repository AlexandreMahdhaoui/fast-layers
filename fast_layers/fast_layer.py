import tensorflow as tf


class _Tracker:
    def __init__(self, inputs, is_output_layer):
        self.inputs = inputs
        self.is_output_layer = is_output_layer
        self.is_computed = False


class Layer(tf.keras.layers.Layer):
    """

    """

    def __init__(self, sequences=None, trainable=True, n_iteration_error=50):
        super(Layer, self).__init__()
        self.names = []
        self.trainable = trainable
        if sequences is not None:
            self.init_layer(sequences)
        self.first_call = True
        self.n_iteration_error = n_iteration_error

    def call(self, x, training=False):
        if self.first_call:
            return self._first_compute_x(x, training=training)
        return self._compute_x(x, training=training)

    def init_layer(self, sequences):
        assert type(sequences) == list, 'sequences must be a list of Sequence objects'
        for item in sequences:
            self.__dict__[item.name] = item
            self.__dict__[item.name + '_tracker'] = _Tracker(item.inputs, item.is_output_layer)
            self.names.append(item.name)
            item.self_build()

    def _compute_x(self, x, training=False):
        self._compute_input_sequences(x, training=training)
        while not self._is_all_sequences_computed():
            for name in self.names:
                self._compute_sequence(name, training=training)
        return self._get_output()

    def _get_output(self):
        for name in self.names:
            if self.__dict__[name + '_tracker'].is_output_layer:
                return self.__dict__[name + '_x']

    def _compute_sequence(self, name, training=False):
        sequence = self.__dict__[name]
        tracker = self.__dict__[name + '_tracker']
        if self._check_inputs(tracker) and not tracker.is_computed:
            x = self._get_x(tracker)
            self.__dict__[name + '_x'] = sequence(x, training=training)
            self.__dict__[name + '_tracker'].is_computed = True

    def _check_inputs(self, tracker):
        if type(tracker.inputs) == str:
            str_ = tracker.inputs + '_x'
            return str_ in self.__dict__.keys()
        else:
            for input_ in tracker.inputs:
                str_ = input_ + '_x'
                if not (str_ in self.__dict__.keys()):
                    return False
                return True

    def _get_x(self, tracker):
        if type(tracker.inputs) == str:
            return self.__dict__[tracker.inputs + '_x']
        else:
            x = []
            for input_ in tracker.inputs:
                x.append(self.__dict__[input_ + '_x'])
            return x

    def _compute_input_sequences(self, x, training=False):
        for name in self.names:
            sequence = self.__dict__[name]
            tracker = self.__dict__[name + '_tracker']
            if tracker.inputs == 'input' and not tracker.is_computed:
                self.__dict__[name + '_x'] = sequence(x, training=training)
                self.__dict__[name + '_tracker'].is_computed = True

    def _is_all_sequences_computed(self):
        for name in self.names:
            if not self.__dict__[name + '_tracker'].is_computed:
                return False
        return True

    def _check_loop_health(self, i):
        if i > self.n_iteration_error:
            print(f'''
            {i}th iterations, please review your fast-layer architecture, the computation of layers might be stucked in an infinite loop.
            Program will terminate.
            If you need more iterations please init FastLayer with a higher value of keyword argument 'n_iteration_error' (base value = 50)
            ''')
            import sys
            sys.exit(f'Infinite loop spotted: program terminated')

    def _first_compute_x(self, x, training=False):
        self._compute_input_sequences(x, training=training)
        i = 0
        while not self._is_all_sequences_computed():
            for name in self.names:
                self._compute_sequence(name, training=training)
            self._check_loop_health(i)
            i += 1
        return self._get_output()


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
            assert (type(inputs) == list and all(type(_) is str for _ in inputs)), \
                'Please either provide a single input as a string, or several inputs as a list of string'
        self.inputs = inputs
        self.is_output_layer = is_output_layer
        if sequence is not None:
            assert (type(sequence) == list), 'Please provide a sequence of one or more layers as a list'
        self.sequence = sequence

    def __call__(self, x, training=False):
        for layer in self.sequence:
            x = layer(x, training=training)
        return x

    def self_build(self):
        for layer in self.sequence:
            self.__dict__[layer.name] = layer
