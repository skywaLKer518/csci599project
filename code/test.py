from ae import *




def go():
	f = file('nnets.save','rb')
	m = cPickle.load(f)
	f.close
	dataset = 'mnist.pkl.gz'

	datasets = load_data(dataset)
	test_set_x, test_set_y = datasets[2]
	
	
	return m,test_set_x,test_set_y

