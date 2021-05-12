import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

dice_types = ['d6']

all_x = np.ones((0, 50, 50), dtype=np.uint8)
all_y = np.ones((0, 1), dtype=np.uint8)

# Import images
for dice in dice_types:
    for face in os.listdir(dice):
        for image_name in os.listdir(f'{dice}\\{face}'):
            image = cv2.imread(f'{dice}\\{face}\\{image_name}')
            image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape((1, 50, 50))
            all_x = np.append(all_x, image_grey, axis=0)
            all_y = np.append(all_y, [[int(face)]], axis=0)

N = all_x.shape[0]
index = np.arange(N)
np.random.shuffle(index)

train_N = int(N * 0.8)

x_train = all_x[index[:train_N]]
y_train = all_y[index[:train_N]]

x_test = all_x[index[train_N:]]
y_test = all_y[index[train_N:]]

# Data Normalization
# Conversion to float
x_train = x_train.astype(np.float32) 
x_test = x_test.astype(np.float32)

# Normalization
x_train = x_train/255.0
x_test = x_test/255.0

# Reshaping input data
X_train = x_train.reshape(len(x_train),-1)
X_test = x_test.reshape(len(x_test),-1)

#total_clusters = len(np.unique(y_test))
clusters = [6, 12, 24, 64, 144]
results = []
for total_clusters in clusters:
    # Initialize the K-Means model
    kmeans = MiniBatchKMeans(n_clusters = total_clusters)
    # Fitting the model to training set
    kmeans.fit(X_train)
    kmeans.labels_
    
    def retrieve_info(cluster_labels, y_train):
        '''
        Associates most probable label with each cluster in KMeans model
        returns: dictionary of clusters assigned to each label
        '''
        # Initializing
        reference_labels = {}
        # For loop to run through each label of cluster label
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(y_train[index==1].reshape(-1)).argmax()
            reference_labels[i] = num
        return reference_labels
    
    reference_labels = retrieve_info(kmeans.labels_,y_train)
    #print(reference_labels)
    number_labels = np.random.rand(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]
        
    # Comparing Predicted values and Actual values
    #print(number_labels[:20].astype('int'))
    #print(y_train[:20])
    
    # Calculating accuracy score
    from sklearn.metrics import accuracy_score
    
    print(accuracy_score(number_labels,y_train))
    
    predicted_cluster = kmeans.predict(X_test)
    
    number_labels = np.zeros(len(predicted_cluster))
    for i in range(len(predicted_cluster)):
        number_labels[i] = reference_labels[predicted_cluster[i]]
    results.append(accuracy_score(number_labels,y_test))
    
plt.plot(clusters, results)
plt.show()