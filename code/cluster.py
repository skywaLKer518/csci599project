# function: K-means
# input data, k

import numpy as np
from numpy import linalg as LA
MAX_ITERATION = 200
EPSILON = 0.00001

def my_kmeans(data,k):
	# initialize the centroids randomly
	d = data.shape[1]
	centroids = getRandomCentroids(data,k)

	# Initialize book keeping vars
	iteration = 0
	oldCentroids = np.zeros((k,d))
	stop = False
	difference = []
	objective = []

	# Run the main k-means algorithm
	while not stop:
		# Save old centroids 
		oldCentroids = centroids
		iteration += 1
#		print('iter: %i')%(iteration)
		# Assige cluster label
		labels,obj = getLabels(data,centroids)
		# Adjust centroids
		centroids = updateCentroids(data,labels,k)
		stop, dif = terminateTest(oldCentroids,centroids,iteration)
		
		difference.append(dif)
		objective.append(obj)
	print('clustering finished at iter: %i.' ''' with objective value:''')%(iteration)
	#print objective
	return centroids,labels,objective

# function : stop test
# ----------
def terminateTest(old,center,t):
	diff = np.linalg.norm(old-center)
	if t > MAX_ITERATION: 
		print 'maximum iteration number reached; exit'
		return True,diff
	if diff < EPSILON:
		return True,diff
	else:
		return False,diff

def getRandomCentroids(data,k):
	ind = np.random.permutation(data.shape[0])[0:k]
	return data[ind,:]
		
# return a label for each data intance
def getLabels(data,centroids):
	xy = np.dot(data,centroids.T)
	x2 = np.tile(LA.norm(data,axis=1),(centroids.shape[0],1))**2
	x2 = x2.T
	y2 = np.tile(LA.norm(centroids,axis=1),(data.shape[0],1))**2
	dis = x2 + y2 - xy * 2  # N1 by k
	y = np.argmin(dis,axis=1)
	dis2 = dis[xrange(data.shape[0]),y]
#	print y
#	print dis
#	print dis2
	return y,dis2.sum()

def updateCentroids(data,label,k):
	center = np.zeros([k,data.shape[1]])
	for i in xrange(k):
		item_ind = np.where(label==i)[0]
		'''print center.shape
		print data.shape
		print item_ind[0].shape
		'''
		'''print i
		print item_ind
		print item_ind[0]
		print data[item_ind[0],:]
		'''
		center[i,] = data[item_ind,:].mean(0)
	return center
def test():
	data = np.zeros((300,5))
	data[0:100,:] = np.random.randn(100,5)*1+5
	data[100:200,:] = np.random.randn(100,5)*3+-1
	data[200:300,:] = np.random.randn(100,5)*1+1
	c,y,his = my_kmeans(data,3)
	return c,y,data

if __name__=='__main__':
	test()	
