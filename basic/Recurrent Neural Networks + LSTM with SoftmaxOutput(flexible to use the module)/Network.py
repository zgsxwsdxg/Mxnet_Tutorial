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
    fc_number=100
    class_number=10
    use_cudnn = True

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
    test_iter   = mx.io.NDArrayIter(data={'data' : test_img}, label={'label' : test_lbl_one_hot}) #test data

    ####################################################-Network-################################################################
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    data = mx.sym.transpose(data, axes=(1, 0, 2))  # (time,batch,column)

    '''1. RNN cell declaration'''

    '''
    Fusing RNN layers across time step into one kernel.
    Improves speed but is less flexible. Currently only
    supported if using cuDNN on GPU.
    '''

    if use_cudnn:#faster
        lstm1 = mx.rnn.FusedRNNCell(num_hidden=hidden_unit_number1, mode="lstm", prefix="lstm1_",get_next_state=True)
        lstm2 = mx.rnn.FusedRNNCell(num_hidden=hidden_unit_number2, mode="lstm", prefix="lstm2_",get_next_state=True)
    else:
        lstm1 = mx.rnn.LSTMCell(num_hidden=hidden_unit_number1, prefix="lstm1_")
        lstm2 = mx.rnn.LSTMCell(num_hidden=hidden_unit_number2, prefix="lstm2_")

    '''2. Unroll the RNN CELL on a time axis.'''

    ''' unroll's return parameter
    outputs : list of Symbol
              output symbols.
    states : Symbol or nested list of Symbol
            has the same structure as begin_state()

    '''
    #if you see the unroll function
    layer1, state1= lstm1.unroll(length=time_step, inputs=data, merge_outputs=True, layout='TNC')
    layer1 = mx.sym.Dropout(layer1, p=0.3)
    layer2, state2 = lstm2.unroll(length=time_step, inputs=layer1, merge_outputs=True,layout="TNC")
    rnn_output= mx.sym.Reshape(state2[-1], shape=(-1,hidden_unit_number1))

    '''FullyConnected Layer'''
    affine1 = mx.sym.FullyConnected(data=rnn_output, num_hidden=fc_number, name='affine1')
    act1 = mx.sym.Activation(data=affine1, act_type='sigmoid', name='sigmoid1')
    affine2 = mx.sym.FullyConnected(data=act1, num_hidden=class_number, name = 'affine2')
    output = mx.sym.SoftmaxOutput(data=affine2, label=label, name='softmax')


    # We visualize the network structure with output size (the batch_size is ignored.)
    shape = {"data": (time_step,batch_size,28)}
    mx.viz.plot_network(symbol=output,shape=shape)#The diagram can be found on the Jupiter notebook.
    print output.list_arguments()

    # training mod

    mod = mx.module.Module(symbol = output , data_names=['data'], label_names=['label'], context=mx.gpu(0))
    mod.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)

    #load the saved mod data
    mod.load_params("weights/Neural_Net-100.params")

    mod.init_params(initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1))
    mod.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001})

    # test mod
    test = mx.mod.Module(symbol=output, data_names=['data'], label_names=['label'], context=mx.gpu(0))

    '''load method1 - using the shared_module'''
    """
    Parameters
    shared_module : Module
        Default is `None`. This is used in bucketing. When not `None`, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
    """
    test.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label,shared_module=mod,for_training=False)

    # Network information print
    print mod.data_names
    print mod.label_names
    print train_iter.provide_data
    print train_iter.provide_label

    '''############Although not required, the following code should be declared.#################'''

    '''make evaluation method 1 - Using existing ones.
        metrics = {
        'acc': Accuracy,
        'accuracy': Accuracy,
        'ce': CrossEntropy,
        'f1': F1,
        'mae': MAE,
        'mse': MSE,
        'rmse': RMSE,
        'top_k_accuracy': TopKAccuracy
    }'''

    metric = mx.metric.create(['acc','mse'])

    '''make evaluation method 2 - Making new things.'''
    '''
    Custom evaluation metric that takes a NDArray function.
    Parameters:
    •feval (callable(label, pred)) – Customized evaluation function.
    •name (str, optional) – The name of the metric.
    •allow_extra_outputs (bool) – If true, the prediction outputs can have extra outputs.
    This is useful in RNN, where the states are also produced in outputs for forwarding.
    '''

    def zero(label, pred):
        return 0

    null = mx.metric.CustomMetric(zero)

    for epoch in xrange(1,epoch+1,1):
        print "epoch : {}".format(epoch)
        train_iter.reset()
        #total_batch_number = np.ceil(len(train_img) / (batch_size * 1.0))
        #temp=0
        for batch in train_iter:
            mod.forward(batch, is_train=True)
            mod.backward()
            mod.update()

            #cost
            #temp+=(mod.get_outputs()[0].asnumpy()-batch.label[0].asnumpy())

        #cost = (0.5*np.square(temp)/(total_batch_number*1.0)).mean()
        result = test.predict(test_iter).asnumpy().argmax(axis=1)
        print "training_data : {}".format(mod.score(train_iter, ['mse', 'acc']))
        print 'accuracy during learning.  : {}%'.format(float(sum(test_lbl == result)) / len(result) * 100.0)
        #print "cost value : {}".format(cost)

        #Save the data
        if epoch%save_period==0:
            print('Saving weights')
            mod.save_params("weights/Neural_Net" .format(epoch))

    # Network information print
    print mod.data_shapes
    print mod.label_shapes
    print mod.output_shapes
    print mod.get_params()
    print mod.get_outputs()

    print "Optimization complete."
    #################################TEST####################################
    '''load method2 - load the training mod.get_params() directly'''
    #arg_params, aux_params = mod.get_params()

    '''Annotate only when running test data. and Uncomment only if it is 'load method2' '''
    #test.set_params(arg_params, aux_params)

    #batch by batch accuracy
    #To use the code below, Test / batchsize must be an integer.
    '''for preds, i_batch, eval_batch in mod.iter_predict(test_iter):
        pred_label = preds[0].asnumpy().argmax(axis=1)
        label = eval_batch.label[0].asnumpy().argmax(axis=1)
        print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label == label)) / len(label)))
    '''
    '''test'''
    result = test.predict(test_iter).asnumpy().argmax(axis=1)
    print 'Final accuracy : {}%' .format(float(sum(test_lbl == result)) / len(result)*100.0)

if __name__ == "__main__":
    print "NeuralNet_starting in main"
    NeuralNet(epoch=100,batch_size=100,save_period=100)
else:
    print "NeuralNet_imported"
