import cv2
import numpy as np


def main():

    seed = 1
    nclusters = 6
    N = 100
    
    np.random.seed(seed) # Get always same random numpys
    data = np.random.random(size=(N, 2)).astype(np.float32) * 100
    centers = np.random.random(size=(nclusters, 2)).astype(np.float32) * 100
    labels = np.random.randint(nclusters,
                               size=(N, 1),
                               dtype=np.int32)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    reshaped_data = data#np.reshape(data, data.shape[0] * data.shape[1])
    reshaped_labels = labels#np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))

    _, new_labels, center = cv2.kmeans(data=reshaped_data,
                                       K=nclusters,
                                       bestLabels=reshaped_labels,
                                       criteria=criteria,
                                       attempts=10,
                                       flags=cv2.KMEANS_USE_INITIAL_LABELS,
                                       centers=centers)
    return new_labels, center


if __name__ == "__main__":
    main()