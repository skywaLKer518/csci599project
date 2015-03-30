from ae import *
from DBN_mm import *
from logistic_sgd import load_data

test_DBN(finetune_lr=0.1,pretraining_epochs=50,pretrain_lr=0.001,k=1,training_epochs=1,
	dataset='grayscale.pkl.gz',batch_size=10,hidden_layers_sizes=[2000],pretrain_model='gray_pre1.save')
test_autoencoder(finetune_lr=0.01,momentum=0.5,training_epochs=30,dataset='grayscale.pkl.gz',batch_size=10, pretrain='gray_pre1.save')

