from ae import *
import numpy as np

HIDDEN = np.array([[300], [1000], [2000], [3000], [5000], [10000]])

# PRE_LR = np.array([0.001, 0.005, 0.008, 0.01, 0.05, 0.1, 0.5], dtype='float32')  # np.array([0.01]) #
# PRE_T = np.array([20])
# PRE_MBS = np.array([50])

# PRE_MODEL = np.array(["output/gray_pre3.save"])
FINE_LR = np.array([0.0001, 0.0005, 0.001, 0.005, 0.008, 0.01, 0.05, 0.1, 0.5, 1], dtype='float32')
FINE_T = np.array([100])  #
FINE_MBS = np.array([50])
LAMBDA = np.array([0, 1, 0.1, 0.01, 0.001], dtype='float32')  #
DATASET = np.array(["grayscale.pkl.gz"])


PRE_MODEL = {'300': 'output/pre_h[300]_lr0.001_t20_mbs50.save',
             '1000': 'output/pre_h[1000]_lr0.008_t20_mbs50.save',
             '2000': 'output/pre_h[2000]_lr0.05_t20_mbs50.save',
             '3000': 'output/pre_h[3000]_lr0.05_t20_mbs50.save',
             '5000': 'output/pre_h[5000]_lr0.05_t20_mbs50.save',
             '10000': 'output/pre_h[10000]_lr0.05_t20_mbs50.save'}


N = HIDDEN.size * FINE_LR.size * FINE_T.size * FINE_MBS.size \
    * DATASET.size * LAMBDA.size
print 'total model number is %d ' % N

dataset = DATASET[0]
fine_mbs = FINE_MBS[0]

for i1 in xrange(HIDDEN.size):
    for i2 in xrange(FINE_LR.size):
        for i3 in xrange(FINE_T.size):
            for i4 in xrange(LAMBDA.size):

                hidden = HIDDEN[i1]
                model = PRE_MODEL[str(hidden[0])]
                fine_lr = FINE_LR[i2]
                fine_t = FINE_T[i3]
                lambda1 = LAMBDA[i4]

                file_saved = 'output/fine_' + 'h' + str(hidden) + \
                             '_lr' + str(fine_lr) + '_t' + str(fine_t) + '_l' + str(lambda1)\
                             + '_mbs' + str(fine_mbs) + '.save'
                file_log = 'log/fine_' + 'h' + str(hidden) + '_lr' \
                           + str(fine_lr) + '_t' + str(fine_t) + '_l' + str(lambda1)\
                           + '_mbs' + str(fine_mbs) + '.txt'
                file_logs = "log/details_fine_" + 'h' + str(hidden) + '_lr' \
                            + str(fine_lr) + '_t' + str(fine_t) + '_l' + str(lambda1)\
                            + '_mbs' + str(fine_mbs) + '.txt'

                print ("\n=== tuning... model name: %s===\n" % file_saved)

                # obj = test_DBN(finetune_lr=0, pretraining_epochs=pre_t,
                # pretrain_lr=pre_lr, k=1, training_epochs=0,
                #                dataset=dataset, batch_size=pre_mbs,
                #                hidden_layers_sizes=hidden,
                #                pretrain_model=file_saved,
                #                logfile=file_logs)

                loss, pred_err, rec_err = test_autoencoder(finetune_lr=fine_lr, momentum=0, lambda1=lambda1,
                                       training_epochs=fine_t, dataset=dataset,
                                       batch_size=fine_mbs, pretrain=model,
                                       model_save=file_saved, log_file=file_logs)

                # output results
                text_file = open(file_log, "w")
                text_file.write("===fine-tuned model=== \nname=%s \n" % file_saved)
                text_file.write("\n===hyperparameters===\n")
                text_file.write("hidden units=%d\n" % hidden)
                text_file.write("pre-train model=%s\n" % model)
                text_file.write("learning rate=%e\n" % fine_lr)
                text_file.write("mini-batch=%d\n" % fine_mbs)
                text_file.write("\nResults:\n")
                text_file.write("Valid loss: %f\n" % loss)
                text_file.write("Valid rec error: %f\n" % rec_err)
                text_file.write("Valid pred error: %f\n" % pred_err)
                text_file.close()


