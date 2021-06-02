# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os, cv2, time, itertools

print(tf.__version__)

#dice_types = ['d6']

## Initialise dataset
#all_x = np.ones((0, 50, 50, 3), dtype=np.uint8)
#all_y = np.ones((1), dtype=np.uint8)

## Import images
#for dice in dice_types:
    #for face in os.listdir(f'{dice}\\train'):
        #for image_name in os.listdir(f'{dice}\\train\\{face}'):
            #image = np.expand_dims(cv2.imread(f'{dice}\\train\\{face}\\{image_name}'), axis=0)
            #all_x = np.append(all_x, image, axis=0)
            #all_y = np.append(all_y, [int(face) - 1])

#N = all_x.shape[0]
#index = np.arange(N)
#np.random.shuffle(index)

## Split data into test and train sets
#train_N = int(N * 0.8)

#train_images = all_x[index[:train_N]]
#train_labels = all_y[index[:train_N]]

#test_images = all_x[index[train_N:]]
#test_labels = all_y[index[train_N:]]

#class_names = ['1', '2', '3', '4', '5', '6']

## Data Normalization
## Conversion to float
#train_images = train_images.astype(np.float32) 
#test_images = test_images.astype(np.float32)

## Normalization
#train_images = (train_images/127.5)-1
#test_images = (test_images/127.5)-1

BATCH_SIZE = 4
IMG_SIZE = (50, 50)

train_dataset = image_dataset_from_directory('d6\\train',
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)
validation_dataset = image_dataset_from_directory('d6\\val',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
class_names = train_dataset.class_names

#plt.figure(figsize=(10, 10))
#for images, labels in train_dataset.take(1):
    #for i in range(9):
        #ax = plt.subplot(3, 3, i + 1)
        #plt.imshow(images[i].numpy().astype("uint8"))
        #plt.title(class_names[labels[i]])
        #plt.axis("off")
#plt.show()

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Artificially add more data
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Rescale pixel values
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV3Small(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')

#base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

inputs = tf.keras.Input(shape=(50, 50, 3))
x = data_augmentation(inputs)
x = rescale(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 1

loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

#plt.figure(figsize=(8, 8))
#plt.subplot(2, 1, 1)
#plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()),1])
#plt.title('Training and Validation Accuracy')

#plt.subplot(2, 1, 2)
#plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.ylabel('Cross Entropy')
#plt.ylim([0,1.0])
#plt.title('Training and Validation Loss')
#plt.xlabel('epoch')
#plt.show()
"""
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)
#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
"""

## https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/
## Convert Keras model to ConcreteFunction
#full_model = tf.function(lambda x: model(x))
#full_model = full_model.get_concrete_function(
    #tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

## Get frozen ConcreteFunction
#frozen_func = convert_variables_to_constants_v2(full_model)
#frozen_func.graph.as_graph_def()

#layers = [op.name for op in frozen_func.graph.get_operations()]
#print("-" * 50)
#print("Frozen model layers: ")
#for layer in layers:
    #print(layer)

#print("-" * 50)
#print("Frozen model inputs: ")
#print(frozen_func.inputs)
#print("Frozen model outputs: ")
#print(frozen_func.outputs)

## Save frozen graph from frozen ConcreteFunction to hard drive
#tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  #logdir="./frozen_models",
                  #name="frozen_graph.pb",
                  #as_text=False)

# initialize TF MobileNet model
original_tf_model = MobileNet(
    include_top=True,
    weights="imagenet"
)
              
# define the directory for .pb model
pb_model_path = "models"
# define the name of .pb model
pb_model_name = "mobilenet.pb"
# create directory for further converted model
os.makedirs(pb_model_path, exist_ok=True)
# get model TF graph
tf_model_graph = tf.function(lambda x: model(x))
# get concrete function
tf_model_graph = tf_model_graph.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
# obtain frozen concrete function
frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)
# get frozen graph
frozen_tf_func.graph.as_graph_def()
# save full tf model
tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                  logdir=pb_model_path,
                  name=pb_model_name,
                  as_text=False)                  