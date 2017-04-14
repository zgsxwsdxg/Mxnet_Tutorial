# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def NeuralNet(epoch,batch_size,save_period):

    '''

    load_data

    1. SoftmaxOutput must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl}, batch_size=batch_size) #test data

    2. LogisticRegressionOutput , LinearRegressionOutput , MakeLoss and so on.. must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data

    '''

    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data

    '''neural network'''
    data = mx.sym.Variable('data')
    data = mx.sym.Flatten(data=data) #Flatten the mnist data
    label = mx.sym.Variable('label')

    # first_hidden_layer
    affine1 = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=50)
    hidden1 = mx.sym.Activation(data=affine1, name='sigmoid1', act_type="sigmoid")

    # two_hidden_layer
    affine2 = mx.sym.FullyConnected(data=hidden1, name='fc2', num_hidden=50)
    hidden2 = mx.sym.Activation(data=affine2, name='sigmoid2', act_type="sigmoid")

    # output_layer
    output_affine = mx.sym.FullyConnected(data=hidden2, name='fc3', num_hidden=10)

    #The soft max label is compatible with both xxxx_lbl and xxxx_lbl_one_hot.
    #output=mx.sym.SoftmaxOutput(data=output_affine,label=label)

    #LogisticRegressionOutput contains a sigmoid function internally. and It should be executed with xxxx_lbl_one_hot data.
    output = mx.sym.LogisticRegressionOutput(data=output_affine ,label=label)

    '''
    cost function  = Mean squaer error(MSE)
    In my opinion, the code below should be executed properly.
    However, if the learning rate is not reduced, it will not work at all.
    '''
    #output = mx.symbol.MakeLoss(mx.sym.square(output_act-label),name="cost")

    print output.list_arguments()

    # Fisrt optimization method
    # weights save

    model_name = 'weights/Neural_Net'
    checkpoint = mx.callback.do_checkpoint(model_name, period=save_period)

    # training mod
    mod = mx.mod.Module(symbol=output, data_names=['data'], label_names=['label'], context=mx.gpu(0))
    # test mod
    test = mx.mod.Module(symbol=output, data_names=['data'], label_names=['label'], context=mx.gpu(0))

    # Network information print
    print mod.data_names
    print mod.label_names
    print train_iter.provide_data
    print train_iter.provide_label

    '''if the below code already is declared by mod.fit function, thus we don't have to write it.
    but, when you load the saved weights, you must write the below code.'''
    mod.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)

    #weights load

    # When you want to load the saved weights, uncomment the code below.
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, 100)

    #the below code needs mod.bind, but If arg_params and aux_params is set in mod.fit, you do not need the code below, nor do you need mod.bind.
    mod.set_params(arg_params, aux_params)


    '''in this code ,  eval_metric, mod.score doesn't work'''
    mod.fit(train_iter, initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=1),
            optimizer='adam',
            optimizer_params={'learning_rate': 0.001},
            eval_metric=mx.metric.MSE(),
            # Once the loaded parameters are declared here,You do not need to declare mod.set_params,mod.bind
            arg_params=None,
            aux_params=None,
            num_epoch=epoch,
            epoch_end_callback=checkpoint)

    # Network information print
    print mod.data_shapes
    print mod.label_shapes
    print mod.output_shapes
    print mod.get_params()
    print mod.get_outputs()
    print mod.score(train_iter, ['mse', 'acc'])

    #################################TEST####################################
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, 100)

    test.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label, for_training=False)

    '''Annotate only when running test data.'''
    test.set_params(arg_params, aux_params)

    #batch by batch accuracy
    #To use the code below, Test / batchsize must be an integer.
    '''for preds, i_batch, eval_batch in mod.iter_predict(test_iter):
        pred_label = preds[0].asnumpy().argmax(axis=1)
        label = eval_batch.label[0].asnumpy().argmax(axis=1)
        print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label == label)) / len(label)))
    '''
    '''all data test'''
    result = test.predict(test_iter).asnumpy().argmax(axis=1)
    print 'Final accuracy : {}%' .format(float(sum(test_lbl == result)) / len(result)*100)

    '''
    #Second optimization method
    model = mx.model.FeedForward(
        symbol=output,  # network structure
        num_epoch=10,  # number of data passes for training
        learning_rate=0.1  # learning rate of SGD
    )
    model.fit(
        X=train_iter,  # training data
        eval_data=test_iter,  # validation data
        batch_end_callback=mx.callback.Speedometer(batch_size, 200)  # output progress for each 200 data batches
    )'''

if __name__ == "__main__":

    print "NeuralNet_starting in main"
    NeuralNet(epoch=100,batch_size=100,save_period=100)

else:

    print "NeuralNet_imported"
