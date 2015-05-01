import matplotlib.pyplot as plt

def compareTwo(x,y):
	plt.figure(1)
	plt.subplot(211)
	plt.imshow(x)
	plt.subplot(212)
	plt.imshow(y)
	plt.show()

def compareTwo2(x,y, size1, size2):
	plt.figure(1)
	plt.subplot(211)
	plt.imshow(x.reshape(size1,size2))
	plt.subplot(212)
	plt.imshow(y.reshape(size1,size2))
	plt.show()

def compareTen(x,size):
	''' given x (n by d) image, plot first k images'''
	plt.figure(1)
	for i in xrange(10):
		plt.subplot(2,5,i)
		plt.imshow(x[i,].reshape(size,size))
	plt.show()

def compareTen2(x,size1,size2):
	''' given x (n by d) image, plot first k images'''

	plt.figure(1)
	for i in xrange(10):
	 	plt.subplot(2, 5, i)
		plt.imshow(x[i, ].reshape(size1, size2))
	plt.show()