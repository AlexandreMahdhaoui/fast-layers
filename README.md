# Fast Layers
Fast Layers is a python library for Keras and Tensorflow users: The fastest way to build complex deep neural network architectures with sequential models



## Introduction
Tensorflow's sequential model is a very intuitive way to start learning about Deep Neural Networks.
However it is quite hard to dive into more complex networks without learning more about Keras.

Well it won't be hard anymore with Fast-layers! Define your Connectors and Pipes to start building complex layers in a sequential fashion.

I create fast-layers for beginners who wants to build more advanced networks and for experimented users who wants to quickly build and test complex module architectures.



## Fast Layer MNIST tutorial but using Inception modules

TRY IT YOURSELF: https://www.kaggle.com/alexandremahdhaoui/fast-layers-tutorial !

original MNIST tutorial: https://www.tensorflow.org/datasets/keras_example

Szegedy et al. 2014, Going deeper with convolutions: https://arxiv.org/pdf/1409.4842.pdf!

![szegedy et al 2014 Inception Module](https://user-images.githubusercontent.com/80970827/112069667-863ff780-8b6c-11eb-8c90-52c3cbc7917a.png)


```python
!pip install fast_layers

# Imports and preprocessing
import fast_layers as fl
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(128)
ds_test = ds_test.batch(128)
```

```python
# Create the inception module with Fast-Layers
N_FILTERS = 16
PADDING = 'same'

inception_module_0 = fl.FastLayer()
inception_module_0.pipes = [
    fl.Pipe('c1', is_input_layer=True, sequential = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING)
    ], output_identifiers=['concat']),
    fl.Pipe('c1_c3', is_input_layer=True, sequential = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (3,3), padding=PADDING)
    ], output_identifiers=['concat']),
    fl.Pipe('c1_c5', is_input_layer=True, sequential = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (5,5), padding=PADDING)
    ], output_identifiers=['concat']),
    fl.Pipe('maxpool3_c1', is_input_layer=True, sequential = [
        tf.keras.layers.Conv2D(N_FILTERS, (3,3), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING)
    ], output_identifiers=['concat']),
]

inception_module_0.connectors = [
    fl.Connector('concat', 4, is_output_layer=True, sequential=[tf.keras.layers.Concatenate(axis=-1)])
]
inception_module_0.build_layer()
```

```python
# Create and train the model
model = tf.keras.models.Sequential([
    inception_module_0,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


history = model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

```
