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

#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# Data Normalization
# Conversion to float
train_images = train_images.astype(np.float32) 
test_images = test_images.astype(np.float32)

# Normalization
train_images = train_images/255.0
test_images = test_images/255.0

#plt.figure(figsize=(10,10))
#for i in range(25):
    #plt.subplot(5,5,i+1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #plt.imshow(train_images[i], cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
#plt.show()

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


"""
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                       100*np.max(predictions_array),
                                class_names[true_label]),
             color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(1, 7))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
"""

'''
# load the input image from disk
image = test_images[0, :, :].reshape((50, 50))

blob = cv2.dnn.blobFromImage(image, 1, (50, 50))

net = cv2.dnn.readNetFromTensorflow(f'frozen_models\\frozen_graph.pb')

net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
idxs = np.argsort(preds[0])[::-1]

# loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
    # draw the top prediction on the input image
    if i == 0:
        text = "Label: {}, {:.2f}%".format(class_names[idx],
                                                   preds[0][idx] * 100)
        cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
    # display the predicted label + associated probability to the
    # console	
    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
                                                            class_names[idx], preds[0][idx]))
# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)'''