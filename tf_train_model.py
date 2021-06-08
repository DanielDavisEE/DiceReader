# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os, cv2, time, itertools

print(tf.__version__)

dice_types = ['d6']

# Initialise dataset
all_x = np.ones((0, 50, 50), dtype=np.uint8)
all_y = np.ones((1), dtype=np.uint8)

# Import images
for dice in dice_types:
    for face in os.listdir(dice):
        for image_name in os.listdir(f'{dice}\\{face}'):
            image = cv2.imread(f'{dice}\\{face}\\{image_name}')
            image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape((1, 50, 50))
            all_x = np.append(all_x, image_grey, axis=0)
            all_y = np.append(all_y, [int(face) - 1])

N = all_x.shape[0]
index = np.arange(N)
np.random.shuffle(index)

# Split data into test and train sets
train_N = int(N * 0.8)

train_images = all_x[index[:train_N]]
train_labels = all_y[index[:train_N]]

test_images = all_x[index[train_N:]]
test_labels = all_y[index[train_N:]]

class_names = ['1', '2', '3', '4', '5', '6']

# Data Normalization
# Conversion to float
train_images = train_images.astype(np.float32) 
test_images = test_images.astype(np.float32)

# Normalization
train_images = train_images/255.0
test_images = test_images/255.0


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(50, 50)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

config = model.get_config()

# https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/
# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)