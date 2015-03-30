from ae import *
from cluster import *
from scipy.stats import mode


def demo_mnist():
	f = file('mnist.save','rb')
	m = cPickle.load(f)
	f.close

	datasets = load_data('mnist.pkl.gz')
	train_set_x, train_set_y = datasets[2]
	y_true = train_set_y.eval()

	xx = m.encode(train_set_x).eval()
	c,yy,diff = my_kmeans(xx,10)

	err = 0
	for i in xrange(10):
		mode_i = mode(y_true[np.where(yy==i)])[0]
		size_i = np.where(yy==i)[0].size
		print ('cluster %i: size: %i, model: %i')%(i,size_i,mode_i)
#		print y_true[np.where(yy==i)]
#		print np.tile(mode_i,(1,size_i))[0]
		err_i = T.neq(y_true[np.where(yy==i)],np.tile(mode_i,(1,size_i))[0]).eval().sum()
		print err_i
		err += err_i	
#		print y_true[np.where(yy==i)]
	print err
	print err * 1.0 / y_true.size
	n = 3	
	#print('labels in cluster %i (first 200)')%(n)
	#print y_true[np.where(yy==3)][0:200]
	n = 7
	#print('labels in cluster %i (first 200)')%(n)
	#print y_true[np.where(yy==7)][0:200]	

if __name__=='__main__':
	demo_mnist()
