from ae import *
from DBN_mm import *


# test_DBN(finetune_lr=0.1,pretraining_epochs=200,pretrain_lr=0.05,k=1,training_epochs=1,
# 	dataset='grayscale.pkl.gz',batch_size=10,hidden_layers_sizes=[2000],pretrain_model='output/gray_pre2.save')

test_autoencoder(finetune_lr=0.1, momentum=0.5, lambda1=0, training_epochs=10000,
                 dataset='grayscale.pkl.gz', batch_size=10, pretrain='output/gray_pre2.save',
                 model_save='output/gray2.save')

