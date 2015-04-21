from ae import *
from DBN_mm import *


# test_DBN(finetune_lr=0.04, pretraining_epochs=20, pretrain_lr=0.06, k=1, training_epochs=1,
# 	dataset='grayscale.pkl.gz', batch_size=50, hidden_layers_sizes=[2000], pretrain_model='output/gray_pre3.save')

test_autoencoder(finetune_lr=0.08, momentum=0.5, lambda1=0, training_epochs=200,
                 dataset='grayscale.pkl.gz', batch_size=50, pretrain='output/gray_pre3.save',
                 model_save='output/gray3.save')

