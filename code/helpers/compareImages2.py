import matplotlib.pyplot as plt

def compareTwo(x,y):
	plt.figure(1)
	plt.subplot(211)
	plt.imshow(x)
	plt.subplot(212)
	plt.imshow(y)
	plt.show()

def compareMore(x,size,k):
	''' given x (n by d) image, plot first k images'''
	plt.figure(1)
	for i in xrange(k):
		n = 210 + i
		plt.subplot(n)
		plt.imshow(x)
	plt.show()
