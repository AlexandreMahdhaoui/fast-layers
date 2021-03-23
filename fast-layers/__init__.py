import tensorflow as tf

class Connector(tf.keras.layers.Layer):
    def __init__(self, identifier, n_inputs, is_input_layer=False, is_output_layer=False, trainable=True, sequential=[], output_identifiers=[]):
        super(Connector, self).__init__()
        self.identifier = identifier
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        self.n_inputs = n_inputs
        self.output_identifiers = output_identifiers # Note that another connector can be appended to self.output_identifiers
        self.is_input_resolved = False
        self.is_computed = False
        self.stored_inputs = []
        self.sequential = sequential
        
        self.stored_inputs = []
        
    # First layer of sequential must be a tf.keras.layers.Concatenate(axis=?) which calls x as a list
    # x.__len__() > 1 because there is not reason to use a connector instead of a pipe in this situation
    def __call__(self, training=False):
        x = self.stored_inputs
        for layer in self.sequential:
            x = layer(x, training=training)
        self.is_computed=True
        self.stored_output = x
    
    def add_output_identifiers(identifier:str):
        self.output_identifiers.append(identifier)
    
    def getOutputShape(self, input_shapes: list):
        self.input_shape = input_shapes
        self.is_input_resolved = True
        self.output_shape = self.compute_output_shape(input_shape)
        return self.output_shape
    
    def setStoredInputs(self, inputs):
        self.stored_inputs.append(inputs)
        if self.stored_inputs.__len__() == self.n_inputs: 
            self.is_input_resolved = True
    

    def self_build(self):
        for layer in self.sequential:
            self.__dict__[layer.name] = layer

class Pipe(tf.keras.layers.Layer):
    def __init__(self, identifier: str, is_input_layer=False, is_output_layer=False, trainable=True, sequential=[], output_identifiers=[]):
        super(Pipe, self).__init__()
        self.identifier = identifier
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        self.trainable=trainable
        self.output_identifiers=output_identifiers # Note that a pipe can be appended in self.output_identifiers only if it takes only this pipe's output as its input. Otherwise use a connector
        self.is_input_resolved = False
        self.is_computed = False
        self.sequential = sequential

    def __call__(self, training=False):
        x = self.stored_inputs
        for layer in self.sequential:
            x = layer(x, training=training)
        self.is_computed=True
        self.stored_output = x
    
    def add_output_identifiers(identifier: str):
        self.output_identifiers.append(identifier)
    
    def buildOutputShape(self, input_shape):
        self.input_shape = input_shape
        self.is_input_resolved = True
        self.output_shape = self.compute_output_shape(input_shape)
        return self.output_shape
    
    def setStoredInputs(self, inputs):
        self.stored_inputs = inputs
        self.is_input_resolved=True
    

    def self_build(self):
        for layer in self.sequential:
            self.__dict__[layer.name] = layer

class FastLayer(tf.keras.layers.Layer):
    def __init__(self, pipes=[], connectors= [], trainable=True, n_iteration_error=50):
        super(FastLayer, self).__init__()
        self.trainable=trainable
        self.identifiers = []
        self.pipes = pipes
        self.connectors = connectors
        
        self.n_iteration_error = n_iteration_error
    
    def call(self, x, training=False):
        return self.compute_x(x, training=training)
    
    def compute_x(self,x, training=False):
        self.compute_input_layers(x, training=training)
        i = 0
        while not self.is_all_layers_computed():
            for identifier in self.identifiers:
                layer = self.__dict__[identifier]
                if layer.is_input_resolved and not layer.is_computed:
                    layer(training=training)
                    self.broadcast_layer(layer)
            self.check_loop_health(i)
            i+=1
        return self.get_output()
    
    def get_output(self):
        for identifier in self.identifiers:
            if self.__dict__[identifier].is_output_layer:
                return self.__dict__[identifier].stored_output
    
       
    def is_all_layers_computed(self):
        condition = True
        for identifier in self.identifiers:
            if not self.__dict__[identifier].is_computed:
                condition = False
        return condition
    
    def compute_input_layers(self, x, training=False):
        for identifier in self.identifiers: 
            layer = self.__dict__[identifier]
            if layer.is_input_layer and not layer.is_computed and not layer.is_input_resolved:
                layer.setStoredInputs(x)
                layer(training=training)
                self.broadcast_layer(layer)
                

    def broadcast_layer(self, layer):
        if layer.output_identifiers.__len__() > 0:
            for identifier in layer.output_identifiers:
                self.__dict__[identifier].setStoredInputs(layer.stored_output)
    
    def build_layer(self):
        for item in self.pipes:
            item.self_build()
            self.__dict__[item.identifier] = item
            self.identifiers.append(item.identifier)
            
        for item in self.connectors:
            item.self_build()
            self.__dict__[item.identifier] = item
            self.identifiers.append(item.identifier)
            
    def check_loop_health(self,i):
        if i > self.n_iteration_error:
            print(f'''
            {i}th iterations, please review your fast-layer architecture computation might be stucked in an infinite loop.
            Program will terminate after {i+2}th iteration.
            If you need more iterations or don't want to print this message please init FastLayer with a higher value of keyword argument 'n_iteration_error' (base value = 50)
            The Program is terminated after n_iteration_error+2
            ''')
        if i > self.n_iteration_error+2:
            import sys 
            sys.exit(f'{i}th iteration: program terminated')


