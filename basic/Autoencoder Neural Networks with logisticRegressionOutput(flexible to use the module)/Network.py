# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

'''unsupervised learning -  Autoencoder'''

def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255.0

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
    train_iter  = mx.io.NDArrayIter(data={'input' : to2d(train_img)},label={'input_' : to2d(train_img)}, batch_size=batch_size, shuffle=True) #training data
    test_iter   = mx.io.NDArrayIter(data={'input' : to2d(test_img)},label={'input_' : to2d(test_img)}) #test data

    '''Autoencoder network

    <structure>
    input - encode - middle - decode -> output
    '''
    input = mx.sym.Variable('input')
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

    #training mod
    mod = mx.mod.Module(symbol=result, data_names=['input'],label_names=['input_'], context=mx.gpu(0))
    mod.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)

    #load the saved mod data
    mod.load_params("Weights/mod-100.params")

    mod.init_params(initializer=mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3))
    mod.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.01})

    #test mod
    test = mx.mod.Module(symbol=result, data_names=['input'],label_names=['input_'], context=mx.gpu(0))
    '''load method2 - using the shared_module'''
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
            #temp+=(mod.get_outputs()[0].asnumpy()-batch.data[0].asnumpy())
            '''
            No need, but must be declared!!!

            in mxnet,If you do not use one of the following two statements, the memory usage becomes 100 percent and the computer crashes.
            It is not necessary for actual calculation, but the above phenomenon does not occur when it is necessary to write it.
            I do not know why. Just think of it as meaningless.

            '''
            '''make evaluation method 1 - Using existing ones'''
            #mod.update_metric(null, batch.label)
            #null.update(batch.label,mod.get_outputs())

            '''make evaluation method 2 - Making new things.'''
            mod.update_metric(metric, batch.label)
            #metric.update(batch.label, mod.get_outputs())

        print "training_data : {}".format(mod.score(train_iter, ['mse']))
        #cost = (0.5*np.square(temp)/(total_batch_number*1.0)).mean()
        #print "cost value : {}".format(cost)

        #Save the data
        if epoch%save_period==0:
            print('Saving weights')
            mod.save_params("Weights/mod-{}.params" .format(epoch))

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

    '''test'''
    result = test.predict(test_iter,num_batch=20).asnumpy()

    '''visualization'''
    column_size=10
    fig ,  ax = plt.subplots(2, column_size, figsize=(column_size, 2))

    for i in xrange(column_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(result[i], (28, 28)),cmap='gray')
        ax[1][i].imshow(np.reshape(result[i+10], (28, 28)),cmap='gray')

    plt.show()

if __name__ == "__main__":

    print "NeuralNet_starting in main"
    NeuralNet(epoch=100,batch_size=100,save_period=100)

else:

    print "NeuralNet_imported"
