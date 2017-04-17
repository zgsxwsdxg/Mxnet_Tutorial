# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)

def NeuralNet(epoch,batch_size,save_period):

    time_step=28
    hidden_unit_number1 = 100
    hidden_unit_number2 = 100
    fullyconnected_unit_number1=100
    class_number=10
    batch_size = 100

    '''
    load_data

    1. SoftmaxOutput must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl}, batch_size=batch_size) #test data
                                                                or
    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data

    2. LogisticRegressionOutput , LinearRegressionOutput , MakeLoss and so on.. must be

    train_iter = mx.io.NDArrayIter(data={'data' : to4d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : to4d(test_img)}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data

    '''
    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : train_img},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'data' : test_img}, label={'label' : test_lbl_one_hot}, batch_size=batch_size) #test data

    ####################################################-Network-################################################################
    use_cudnn=True

    data = mx.sym.Variable('data')
    label = mx.sym.Variable(('label'))
    data = mx.sym.transpose(data, axes=(1, 0, 2))  # (time,batch,column)

    '''1. RNN cell declaration'''

    '''
    Fusing RNN layers across time step into one kernel.
    Improves speed but is less flexible. Currently only
    supported if using cuDNN on GPU.
    '''

    if use_cudnn: #faster!!!
        rnn1 = mx.rnn.FusedRNNCell(num_hidden=hidden_unit_number1, mode="rnn_tanh", prefix="rnn1_",get_next_state=True)
        rnn2 = mx.rnn.FusedRNNCell(num_hidden=hidden_unit_number2, mode="rnn_tanh", prefix="rnn2_",get_next_state=True)
    else:
        rnn1 = mx.rnn.RNNCell(num_hidden=hidden_unit_number1,activation='tanh', prefix='rnn1_')
        rnn2 = mx.rnn.RNNCell(num_hidden=hidden_unit_number2,activation='tanh', prefix='rnn2_')


    '''2. Unroll the RNN CELL on a time axis.'''


    ''' unroll's return parameter
    outputs : list of Symbol
              output symbols.
    states : Symbol or nested list of Symbol
            has the same structure as begin_state()

    '''
    layer1, state1= rnn1.unroll(length=time_step, inputs=data, merge_outputs=True, layout='TNC')
    layer1 = mx.sym.Dropout(layer1, p=0.3)
    layer2, state2 = rnn2.unroll(length=time_step, inputs=layer1, merge_outputs=True,layout="TNC")
    rnn_output= mx.sym.Reshape(state2[-1], shape=(-1,hidden_unit_number1))

    '''FullyConnected Layer'''
    affine1 = mx.sym.FullyConnected(data=rnn_output, num_hidden=fullyconnected_unit_number1, name='affine1')
    act1 = mx.sym.Activation(data=affine1, act_type='sigmoid', name='sigmoid1')
    affine2 = mx.sym.FullyConnected(data=act1, num_hidden=class_number, name = 'affine2')
    output = mx.sym.SoftmaxOutput(data=affine2, label=label, name='softmax')

    print output.list_arguments()

    # training mod
    mod = mx.module.Module(symbol = output , data_names=['data'], label_names=['label'], context=mx.gpu(0))
    # test mod
    test = mx.module.Module(symbol = output , data_names=['data'], label_names=['label'], context=mx.gpu(0))

    # Network information print
    print mod.data_names
    print mod.label_names
    print train_iter.provide_data
    print train_iter.provide_label

    '''if the below code already is declared by mod.fit function, thus we don't have to write it.
    but, when you load the saved weights, you must write the below code.'''
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    # weights save

    model_name = 'weights/Neural_Net'
    checkpoint = mx.callback.do_checkpoint(model_name, period=save_period)

    #weights load

    # When you want to load the saved weights, uncomment the code below.
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, 300)

    #the below code needs mod.bind, but If arg_params and aux_params is set in mod.fit, you do not need the code below, nor do you need mod.bind.
    mod.set_params(arg_params, aux_params)

    '''in this code ,  eval_metric, mod.score doesn't work'''
    mod.fit(train_iter, initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=1),
            optimizer='adam',
            optimizer_params={'learning_rate': 0.0001},
            eval_metric=mx.metric.MSE(),
            # Once the loaded parameters are declared here,You do not need to declare mod.set_params,mod.bind
            num_epoch=epoch,
            arg_params=None,
            aux_params=None,
            epoch_end_callback=checkpoint)


    # Network information print
    print mod.data_shapes
    print mod.label_shapes
    print mod.output_shapes
    print mod.get_params()
    print mod.get_outputs()
    print mod.score(train_iter, ['mse', 'acc'])

    #################################TEST####################################
    #symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, 300)
    arg_params, aux_params = mod.get_params()

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

if __name__ == "__main__":

    print "NeuralNet_starting in main"
    NeuralNet(epoch=300,batch_size=100,save_period=100)

else:

    print "NeuralNet_imported"
