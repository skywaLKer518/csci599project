from ae import *
from DBN_mm import *
from logistic_sgd import load_data

test_DBN(finetune_lr=0.1,pretraining_epochs=500,pretrain_lr=0.001,k=1,training_epochs=1,
	dataset='grayscale.pkl.gz',batch_size=10,hidden_layers_sizes=[1000],pretrain_model='output/gray_pre2.save')
test_autoencoder(finetune_lr=0.03,momentum=0.5,training_epochs=100,dataset='grayscale.pkl.gz',batch_size=10, pretrain='output/gray_pre2.save',model_save='output/gray2.save')

