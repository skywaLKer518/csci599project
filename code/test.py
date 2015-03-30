from helpers.compareImages import *
from logistic_sgd import load_data
from cluster import my_kmeans

datasets = load_data('grayscale.pkl.gz')
x_tr, y_tr = datasets[0]
x_tr = x_tr.eval()

c,y,obj = my_kmeans(x_tr,20)

compareTen(c,56)
