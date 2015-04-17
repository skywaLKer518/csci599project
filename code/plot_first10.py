'''
A sample function to plot first 10 images
'''

from helpers.compareImages import *
from data_preprocess import load_data
import cPickle

# load data
ds = 'grayscale.pkl.gz'
data_sets = load_data(ds)
x1, y1 = data_sets[0] # train
x2, y2 = data_sets[1] # valid
xmean = data_sets[3]

x1 = (x1 ).eval()
x2 = (x2 ).eval()

print x1.shape
print x1.max()
print x2.min()
# load model
f = file('output/gray2.save', 'rb')
m = cPickle.load(f)
f.close()

# reconstruct
x_rec = (m.reconstruct(x1) ).eval()


# plot first ten images
compareTen2(x_rec, 26, 56)
