# function: K-means
# input data, k

import numpy as np
from numpy import linalg as LA
MAX_ITERATION = 100

def kmeans(data,k):
	# initialize the centroids randomly
	d = data.shape(1)
	centroids = getRandomCentroids(data,k)

	# Initialize book keeping vars
	iteration = 0
	oldCentroids = None

	# Run the main k-means algorithm
	while not terminateTest(oldCentroids,centroids,iteration)
		# Save old centroids 
		oldCentroids = centroids
		iteration += 1

		# Assige cluster label
		labels = getLabel(data,centroids)

		# Adjust centroids
		centroids = updateCentroids(data,labels,k)
	return centroids,label

# function : stop test
# ----------
def terminateTest(old,center,t):
	if t > MAX_ITERATION: 
		return True
	return old == center

def getRandomCentroids(data,k):
	ind = np.random.permutation(data.shape[0])[0:k]
	return data[ind,:]
		
# return a label for each data intance
def getLabels(data,centroids):
	xy = np.dot(data,centroids.T)
	x2 = np.tile(LA.norm(data,axis=1),(centroids.shape[0],1))**2.T
	y2 = np.tile(LA.norm(centroids,axis=1),(data.shape[0],1))**2
	dis = x2 + y2 - xy * 2  # N1 by N2
	return np.argmin(dis,axis=1)

		
