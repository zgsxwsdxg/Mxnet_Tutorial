# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

'''unsupervised learning -  Autoencoder'''

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255

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

    '''In this Autoencoder tutorial, we don't need the label data.'''
    (_, _, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (_, _, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter  = mx.io.NDArrayIter(data={'input' : to4d(train_img)},label={'input_' : to2d(train_img)}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'input' : to4d(test_img)},label={'input_' : to2d(test_img)} ,batch_size=batch_size) #test data

    '''Autoencoder network

    <structure>
    input - encode - middle - decode -> output
    '''
    input = mx.sym.Variable('input')

    input = mx.sym.Flatten(data=input) #Flatten the mnist data
    output= mx.sym.Variable('input_')

    # encode
    affine1 = mx.sym.FullyConnected(data=input,name='encode',num_hidden=100)
    encode1 = mx.sym.Activation(data=affine1, name='sigmoid1', act_type="sigmoid")

    # middle
    affine2 = mx.sym.FullyConnected(data=encode1, name='middle', num_hidden=50)
    middle = mx.sym.Activation(data=affine2, name='sigmoid2', act_type="sigmoid")

    # decode
    affine3 = mx.sym.FullyConnected(data=middle,name='decode',num_hidden=100)
    decode1 = mx.sym.Activation(data=affine3, name='sigmoid1', act_type="sigmoid")

    # output
    result = mx.sym.FullyConnected(data=decode1, name='result', num_hidden=784)

    #LogisticRegressionOutput contains a sigmoid function internally. and It should be executed with xxxx_lbl_one_hot data.
    result = mx.sym.LogisticRegressionOutput(data=result ,label=output)

    print result.list_arguments()

    # Fisrt optimization method
    # weights save

    model_name = 'weights/Autoencoder'
    checkpoint = mx.callback.do_checkpoint(model_name, period=save_period)

    #training mod
    mod = mx.mod.Module(symbol=result, data_names=['input'],label_names=['input_'], context=mx.gpu(0))

    #test mod
    test = mx.mod.Module(symbol=result, data_names=['input'],label_names=['input_'], context=mx.gpu(0))

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

    '''if you want to modify the learning process, go into the mod.fit function()'''

    mod.fit(train_iter, initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=1),
            optimizer='adam', #optimizer
            optimizer_params={'learning_rate': 0.001}, #learning rate
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

    print "completed"


    #################################TEST####################################
    #symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, 100)
    arg_params, aux_params = mod.get_params()

    test.bind(data_shapes=test_iter.provide_data,label_shapes=test_iter.provide_label,for_training=False)

    '''Annotate only when running test data.'''
    test.set_params(arg_params, aux_params)

    '''all data test'''
    result = test.predict(test_iter).asnumpy()

    '''visualization'''
    print_size=10
    fig ,  ax = plt.subplots(2, print_size, figsize=(print_size, 2))

    for i in xrange(print_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(result[i], (28, 28)))
        ax[1][i].imshow(np.reshape(result[i+10], (28, 28)))

    plt.show()

if __name__ == "__main__":

    print "NeuralNet_starting in main"
    NeuralNet(epoch=100,batch_size=100,save_period=10)

else:

    print "NeuralNet_imported"
