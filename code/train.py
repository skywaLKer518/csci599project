from ae import *
from DBN_mm import *


test_DBN(finetune_lr=0.008, pretraining_epochs=20, pretrain_lr=0.008, k=1, training_epochs=1,
	dataset='grayscale_seg_binary_data.pkl.gz', batch_size=50, hidden_layers_sizes=[1000], pretrain_model='output/gray_seg_binary.save')

test_autoencoder(finetune_lr=0.05, momentum=0.5, lambda1=10, training_epochs=60,
                 dataset='grayscale_seg_binary_data.pkl.gz', batch_size=50, pretrain='output/gray_seg_binary.save',
                 model_save='output/gray_seg_binary.save', log_file='binary')

