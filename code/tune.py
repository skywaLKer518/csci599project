from ae import *
from DBN_mm import *
import numpy as np
import sys

HIDDEN = np.array([[100], [500], [1000], [2000]]) # [3000], [5000], [10000]])  # np.array([[400]])

PRE_LR = np.array([0.001, 0.005, 0.01, 0.1, 0.5], dtype='float32')  # np.array([0.01]) #
PRE_T = np.array([20])
PRE_MBS = np.array([50])

# FINE_LR = np.array([0.001, 0.005, 0.008, 0.01, 0.05, 0.1, 0.5, 1])
# FINE_T = np.array([10, 40, 200])

DATASET = np.array(["grayscale_seg_binary_data.pkl.gz"])

N = HIDDEN.size * PRE_LR.size * PRE_T.size * PRE_MBS.size \
    * DATASET.size
# * FINE_LR.size * FINE_T.size * DATASET.size
# print N

dataset = DATASET[0]

for i1 in xrange(HIDDEN.size):
    for i2 in xrange(PRE_LR.size):
        for i3 in xrange(PRE_T.size):
            for i4 in xrange(PRE_MBS.size):
                hidden = HIDDEN[i1]
                pre_lr = PRE_LR[i2]
                pre_t = PRE_T[i3]
                pre_mbs = PRE_MBS[i4]
                file_saved = 'output/pre_' + 'h' + str(hidden) + '_lr' \
                             + str(pre_lr) + '_t' + str(pre_t) \
                             + '_mbs' + str(pre_mbs) + '.save'
                file_log = 'log/pre_' + 'h' + str(hidden) + '_lr' \
                           + str(pre_lr) + '_t' + str(pre_t) \
                           + '_mbs' + str(pre_mbs) + '.txt'
                file_logs = "log/details_pre_" + 'h' + str(hidden) + '_lr' \
                            + str(pre_lr) + '_t' + str(pre_t) \
                            + '_mbs' + str(pre_mbs) + '.txt'

                print ("\n=== tuning... model name: %s===\n" % file_saved)

                obj = test_DBN(finetune_lr=0, pretraining_epochs=pre_t,
                               pretrain_lr=pre_lr, k=1, training_epochs=0,
                               dataset=dataset, batch_size=pre_mbs,
                               hidden_layers_sizes=hidden,
                               pretrain_model=file_saved,
                               logfile=file_logs)
                # output results
                text_file = open(file_log, "w")
                text_file.write("===pre-trained model=== \nname=%s \n" % file_saved)
                text_file.write("\n===hyperparameters===\n")
                text_file.write("hidden units=%d\n" % hidden)
                text_file.write("learning rate=%e\n" % pre_lr)
                text_file.write("mini-batch=%d\n" % pre_mbs)
                text_file.write("\nResults:\n")
                text_file.write("Best cost: %f\n" % obj)
                text_file.close()
#
# test_autoencoder(finetune_lr=0.1, momentum=0.5, lambda1=0, training_epochs=5,
# dataset='grayscale.pkl.gz', batch_size=50, pretrain='output/gray_pre2.save',
# model_save='output/gray2.save')

