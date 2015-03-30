import matplotlib.pyplot as plt

def compareTwo(x,y):
	plt.figure(1)
	plt.subplot(211)
	plt.imshow(x)
	plt.subplot(212)
	plt.imshow(y)
	plt.show()
