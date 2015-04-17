from ae import *
from DBN_mm import *


test_DBN(finetune_lr=0.01, pretraining_epochs=10, pretrain_lr=0.06, k=1, training_epochs=1,
	dataset='grayscale.pkl.gz', batch_size=50, hidden_layers_sizes=[2000], pretrain_model='output/gray_pre2.save')

test_autoencoder(finetune_lr=0.1, momentum=0.5, lambda1=0, training_epochs=5,
                 dataset='grayscale.pkl.gz', batch_size=50, pretrain='output/gray_pre2.save',
                 model_save='output/gray2.save')

