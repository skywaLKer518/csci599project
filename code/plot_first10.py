'''
A sample function to plot first 10 images
'''

from ae import *
from helpers.compareImages import *
from data_preprocess import load_data
import cPickle

# load data
ds = 'grayscale_seg.pkl.gz'
ds = 'rgb_seg_binary_data.pkl.gz'
ds = 'grayscale_seg_binary_data.pkl.gz'
data_sets = load_data(ds)
x1, y1 = data_sets[0] # train
x2, y2 = data_sets[1] # valid
xmean = data_sets[3]

print y1.eval().min()
print y1.eval().max()

x1 = (x1 ).eval()
x2 = (x2 ).eval()

x1 = x1[:, 0:64*48];
x2 = x2[:, 0:64 * 48];
print x1.shape
print x1.max()
print x2.min()
# load model
# f = file('output/gray3.save','rb')
# f = file('output/gray_seg1.save', 'rb')
# m = cPickle.load(f)
# f.close()

# reconstruct
# x_rec = (m.reconstruct(x2) ).eval()


# plot first ten images
# compareTen2(x2, 64, 96)
compareTen2(x1, 48, 64)
n = 5
# compareTwo2(x1[n, ], x_rec[n, ], 26, 56)