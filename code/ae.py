"""
"""
import os
import sys
import time
import numpy
import cPickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression
from data_preprocess import load_data
from DBN import DBN
from mlp import HiddenLayer
# from rbm import RBM

class AutoEncoder(object):
    """ AutoEncoder to do dimension reduction
    """
    def __init__(self,input,numpy_rng,rbm_layers=None, n_layers=None,
                 theano_rng=None, n_ins=6144, n_outs=5):  # 26*56 , 4
        self.hidden_layers = []
        self.params = []
        self.n_layers = n_layers * 2

        # print self.n_layers

        assert self.n_layers > 0
        self.rbm_layers = rbm_layers # for debug...
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        if input is not None:
            self.x, self.y = input
        else:
            self.x = T.matrix('x')
            self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            # construct mlp layers
            if i == 0:
                input_size = n_ins
                output_size = self.rbm_layers[0].n_hidden
                input_w = theano.shared(
                        value=self.rbm_layers[0].W.get_value(),
                        name='W',
                        borrow=True
                )
                input_b = theano.shared(
                        value=self.rbm_layers[0].hbias.get_value(),
                        name='b',
                        borrow=True
                )
            else:
                if i + 1 <= n_layers:
                    input_size = self.rbm_layers[i].n_visible
                    output_size = self.rbm_layers[i].n_hidden
                    input_w = theano.shared(
                        value=self.rbm_layers[i].W.get_value(),
                        name='W',
                        borrow=True
                    )
                    input_b = theano.shared(
                        value=self.rbm_layers[i].hbias.get_value(),
                        name='b',
                        borrow=True
                    )
                else:
                    i_ind = self.n_layers-1-i
                    input_size = self.rbm_layers[i_ind].n_hidden
                    output_size = self.rbm_layers[i_ind].n_visible
                    input_w = theano.shared(
                        value=self.rbm_layers[i_ind].W.get_value().T,
                        name='W',
                        borrow=True
                    )
                    input_b = theano.shared(
                        value=self.rbm_layers[i_ind].vbias.get_value(),
                        name='b',
                        borrow=True
                    )

            # print ('input size for %i layer: %i'%(i, input_size))
            # print ('output size for %i layer: %i'%(i,output_size))
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.hidden_layers[-1].output

            hidden_layer = HiddenLayer(rng=numpy_rng,
                                    input=layer_input,
                                    n_in=input_size,
                                    n_out=output_size,
                                    W=input_w,
                                    b=input_b,
                                    activation=T.nnet.sigmoid)

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

        self.logLayer = LogisticRegression(
            input=self.hidden_layers[n_layers - 1].output,
            n_in=self.rbm_layers[n_layers - 1].n_hidden,
            n_out=n_outs
        )
        self.params.extend(self.logLayer.params)
        self.x_rec = self.hidden_layers[-1].output

    def encode(self,x_in):
        if x_in is None:
            x_in = self.x
        out = x_in
        for i in xrange(self.n_layers/2):
            out = T.nnet.sigmoid(T.dot(out,self.hidden_layers[i].W)+self.hidden_layers[i].b)
        return out

    def reconstruct(self,x_in):
        if x_in is None:
            x_in = self.x
        out = x_in
        for i in xrange(self.n_layers):
            out = T.nnet.sigmoid(T.dot(out,self.hidden_layers[i].W)+self.hidden_layers[i].b)
        return out

    def get_rec_cost(self, x_rec):
        # print T.mean(((self.x-x_rec)**2)).shape.eval()
        return T.mean(((self.x - x_rec) ** 2))
        # return T.mean([2,2])
        # return T.mean(((self.x - x_rec)**2).sum(axis=1))

    def auto_encoder_cost(self, lambda_supervise=1):
        return self.get_rec_cost(self.x_rec) + lambda_supervise * self.logLayer.negative_log_likelihood(self.y)
        # print self.hidden_layers[-1].output.shape.eval()
        # return self.get_rec_cost(self.hidden_layers[-1].output)
        # return self.get_rec_cost(self.x)

    def build_finetune_functions(self,datasets,batch_size,learning_rate,
            lambda_supervise=1,momentum=0.1):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        # print n_valid_batches
        # print n_test_batches

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradient
        cost = self.auto_encoder_cost()
        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params,gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.auto_encoder_cost(lambda_supervise),
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size:(index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size:(index + 1) * batch_size
                ]
            }
        )

        train_score_i2 = theano.function(
            [index],
            self.logLayer.errors(self.y),
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )
        test_score_i = theano.function(
            [index],
            outputs= self.auto_encoder_cost(lambda_supervise),
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_score_i1 = theano.function(
            inputs=[index],
            outputs=self.get_rec_cost(self.x_rec),
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )
        test_score_i2 = theano.function(
            [index],
            self.logLayer.errors(self.y),
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )
        valid_score_i = theano.function(
            [index],
            self.auto_encoder_cost(lambda_supervise),
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )
        valid_score_i1 = theano.function(
            [index],
            self.get_rec_cost(self.x_rec),
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )
        valid_score_i2 = theano.function(
            [index],
            self.logLayer.errors(self.y),
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i3 = theano.function(
            [index],
            self.logLayer.negative_log_likelihood(self.y),
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        def train_score2():
            return [train_score_i2(i) for i in xrange(n_train_batches)]

        def valid_score():  # total loss
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        def valid_score1():  # rec
            return [valid_score_i1(i) for i in xrange(n_valid_batches)]

        def valid_score2():  # error
            return [valid_score_i2(i) for i in xrange(n_valid_batches)]

        def valid_score3():  # neg log likelihood
            return [valid_score_i3(i) for i in xrange(n_valid_batches)]

        def test_score1():
            return [test_score_i1(i) for i in xrange(n_test_batches)]

        def test_score2():
            return [test_score_i2(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score, valid_score1, valid_score2,\
               valid_score3, test_score1, test_score2, train_score2

def test_autoencoder(finetune_lr=0.05, momentum=0.5, lambda1=1, training_epochs=30,
                     dataset='grayscale.pkl.gz', batch_size=10,
                     pretrain='output/gray_pre.save', model_save='output/gray.save', log_file="newLog2"):
    """
    Take pre-trained models as input. Fold the network and fine-tune weights.
    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    :type
    :type momentum: float
    :param momentum
    """

    logfile = open(log_file, "w")
    print >> logfile, "loading data"
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    x_mean = datasets[3]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)

    # load trained model
    print >> logfile, 'loading the model'
    f = file(pretrain, 'rb')
    s_rbm = cPickle.load(f)
    f.close()
    # s_rbm.rbm_layers
    n_layers_rbm = s_rbm.n_layers
    # print >> logfile, n_layers_rbm

    bb = AutoEncoder(None, numpy_rng, s_rbm.rbm_layers, n_layers_rbm)

    print >> logfile, 'getting the fine-tuning functions'
    train_fn, validate_model, test_model, v_m1, v_m2, v_m3, t_m1, t_m2, tr_m2 = bb.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr,
        lambda_supervise=lambda1,
        momentum=momentum
    )

    print >> logfile, '... fine-tuning the model'
    # early-stopping parameters
    patience = 50 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                                # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    # print >> logfile, n_train_batches
    # print >> logfile, patience, patience_increase, validation_frequency, best_validation_loss

    while epoch < training_epochs and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                this_train_loss = numpy.mean(minibatch_avg_cost)

                validation_losses = validate_model()
                validation_rec_error = v_m1()
                validation_error = v_m2()
                validation_neglog = v_m3()

                this_validation_loss = numpy.mean(validation_losses)
                this_validation_rec_error = numpy.mean(validation_rec_error)
                this_validation_error = numpy.mean(validation_error)
                this_validation_neglog = numpy.mean(validation_neglog)
                print >> logfile, (
                        'epoch %i, minibatch %i/%i, train/valid loss %f / %f \n\t\t(rec_error: %f, pred neg log: %f pred_error: %f)'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_train_loss,
                        this_validation_loss,
                        this_validation_rec_error,
                        this_validation_neglog,
                        this_validation_error
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    model_validation_error = this_validation_error
                    model_rec_error = this_validation_rec_error
                    best_iter = iter

                    # test it on the test set
                    test_err = t_m2()
                    test_error = numpy.mean(test_err)

                    # and on the train set
                    train_err = tr_m2()
                    train_error = numpy.mean(train_err)

                    print >> logfile, (('     epoch %i, minibatch %i/%i, train/test error of '
                            'best model %f/%f ') %
                            (epoch, minibatch_index + 1, n_train_batches,
                            train_error, test_error))

            if patience <= iter:
                done_looping = True
                break
    logfile.close()
    f = file(model_save, 'wb')
    cPickle.dump(bb, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return best_validation_loss, model_validation_error, model_rec_error
if __name__ == '__main__':
    test_autoencoder()
