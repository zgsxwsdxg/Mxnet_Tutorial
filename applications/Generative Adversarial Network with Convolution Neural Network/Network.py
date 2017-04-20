# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

'''unsupervised learning -  Generative Adversarial Networks'''
def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255.0

class NoiseIter(mx.io.DataIter):

    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('noise', (batch_size, ndim))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim),ctx=mx.gpu(0))]

def Data_Processing(batch_size):

    '''In this Gan tutorial, we don't need the label data.'''
    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data': to2d(train_img)}, batch_size=batch_size, shuffle=True)  # training data
    return train_iter,len(train_img)

def Generator():

    #generator neural networks
    noise = mx.sym.Variable('noise') # The size of noise is 128.
    g_affine1 = mx.sym.FullyConnected(data=noise, name='g_affine1', num_hidden=256)
    generator1= mx.sym.Activation(data=g_affine1, name='g_sigmoid1', act_type='sigmoid')
    g_affine2 = mx.sym.FullyConnected(data=generator1, name='g_affine2', num_hidden=784)
    g_out= mx.sym.Activation(data=g_affine2, name='g_sigmoid2', act_type='sigmoid')
    return g_out

def Discriminator():

    #discriminator neural networks
    data = mx.sym.Variable('data') # The size of data is 784(28*28)
    d_affine1 = mx.sym.FullyConnected(data=data,name = 'd_affine1' , num_hidden=256)
    discriminator1 = mx.sym.Activation(data=d_affine1, name='d_sigmoid1', act_type='sigmoid')
    d_affine2 = mx.sym.FullyConnected(data=discriminator1,name = 'd_affine2' , num_hidden=128)
    discriminator2 = mx.sym.Activation(data=d_affine2, name='d_sigmoid2', act_type='sigmoid')
    d_affine3 = mx.sym.FullyConnected(data=discriminator2, name='d_affine3', num_hidden=1)
    d_out = mx.sym.Activation(data=d_affine3, name='d_sigmoid3', act_type='sigmoid')

    '''expression-1'''
    #out1 = mx.sym.MakeLoss(mx.symbol.log(d_out),grad_scale=-1.0,normalization='batch',name="loss1")
    #out2 = mx.sym.MakeLoss(mx.symbol.log(1.0-d_out),grad_scale=-1.0,normalization='batch',name='loss2')

    '''expression-2,
    question? Why multiply the loss equation by -1?
    answer : for Maximizing the Loss function , and This is because mxnet only provides optimization techniques that minimize.
    '''
    out1 = mx.sym.MakeLoss(-1.0*mx.symbol.log(d_out),grad_scale=1.0,normalization='null',name="loss1")
    out2 = mx.sym.MakeLoss(-1.0*mx.symbol.log(1.0-d_out),grad_scale=1.0,normalization='null',name='loss2')

    group=mx.sym.Group([out1,out2])

    return group

def GAN(epoch,noise_size,batch_size,save_period):

    train_iter,train_data_number= Data_Processing(batch_size)
    noise_iter = NoiseIter(batch_size, noise_size)

    #No need, but must be declared.
    label =mx.nd.zeros((batch_size,))

    column_size=10
    row_size=2

    '''
    Generative Adversarial Networks

    <structure>
    generator(size = 128) - 256 - (size = 784 : image generate)

    discriminator(size = 784) - 256 - 128 - (size=1 : Identifies whether the image is an actual image or not)

    cost_function - MIN_MAX cost_function
    '''
    '''Network'''

    generator=Generator()
    discriminator=Discriminator()


    '''In the code below, the 'inputs_need_grad' parameter in the 'mod.bind' function is very important.'''

    # =============module G=============
    modG = mx.mod.Module(symbol=generator, data_names=['noise'], label_names=None, context= mx.gpu(0))
    modG.bind(data_shapes=noise_iter.provide_data,label_shapes=None,for_training=True)

    #load the saved modG data
    modG.load_params("Weights/modG-100.params")

    modG.init_params(initializer=mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3))
    modG.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.01})


    # =============module discriminator[0],discriminator[1]=============
    modD_0 = mx.mod.Module(symbol=discriminator[0], data_names=['data'], label_names=None, context= mx.gpu(0))
    modD_0.bind(data_shapes=train_iter.provide_data,label_shapes=None,for_training=True,inputs_need_grad=True)

    # load the saved modD_O data
    modD_0.load_params("Weights/modD_0-100.params")

    modD_0.init_params(initializer=mx.initializer.Xavier(rnd_type='uniform', factor_type='avg', magnitude=3))
    modD_0.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.01})

    """
    Parameters
    shared_module : Module
        Default is `None`. This is used in bucketing. When not `None`, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).

    In here, for sharing the Discriminator parameters, we must to use shared_module=modD_0
    """
    modD_1 = mx.mod.Module(symbol=discriminator[1], data_names=['data'], label_names=None, context= mx.gpu(0))
    modD_1.bind(data_shapes=train_iter.provide_data,label_shapes=None,for_training=True,inputs_need_grad=True,shared_module=modD_0)

    # =============generate image=============
    test_mod = mx.mod.Module(symbol=generator, data_names=['noise'], label_names=None, context= mx.gpu(0))


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

    ####################################training loop############################################
    # =============train===============
    for epoch in xrange(1,epoch+1,1):
        Max_cost_0=0
        Max_cost_1=0
        Min_cost=0
        total_batch_number = np.ceil(train_data_number/(batch_size*1.0))
        print "epoch : {}".format(epoch)
        train_iter.reset()
        for batch in train_iter:
            ################################updating only parameters related to modD.########################################
            # updating discriminator on real data
            '''MAX : modD_0 : -mx.symbol.log(discriminator2)  real data Discriminator update , bigger and bigger discriminator2'''
            modD_0.forward(batch, is_train=True)
            modD_0.backward()
            modD_0.update()

            '''Max_Cost of real data Discriminator'''
            Max_cost_0-=modD_0.get_outputs()[0].asnumpy()

            # update discriminator on noise data
            '''MAX : modD_1 :-mx.symbol.log(1-discriminator2)  - noise data Discriminator update , bigger and bigger -> smaller and smaller discriminator2'''
            noise = noise_iter.next()
            modG.forward(noise, is_train=True)
            modG_output = modG.get_outputs()

            modD_1.forward(mx.io.DataBatch(modG_output, None), is_train=True)
            modD_1.backward()
            modD_1.update()

            '''Max_Cost of noise data Discriminator'''
            Max_cost_1-=modD_0.get_outputs()[0].asnumpy()

            ################################updating only parameters related to modG.########################################
            # update generator on noise data
            '''MIN : modD_0 : -mx.symbol.log(discriminator2) - noise data Discriminator update  , bigger and bigger discriminator2'''
            modD_0.forward(mx.io.DataBatch(modG_output, None), is_train=True)
            modD_0.backward()
            diff_v = modD_0.get_input_grads()
            modG.backward(diff_v)
            modG.update()

            '''Max_Cost of noise data Generator'''
            Min_cost-=modG.get_outputs()[0].asnumpy()

            '''
            No need, but must be declared!!!

            in mxnet,If you do not use one of the following two statements, the memory usage becomes 100 percent and the computer crashes.
            It is not necessary for actual calculation, but the above phenomenon does not occur when it is necessary to write it.
            I do not know why. Just think of it as meaningless.

            '''
            '''make evaluation method 1 - Using existing ones'''
            #metric.update([label], modD_0.get_outputs())
            '''make evaluation method 2 - Making new things.'''
            null.update([label], modD_0.get_outputs())


        Max_C=(Max_cost_0+Max_cost_1)/(total_batch_number*1.0)
        Min_C=Max_cost_0/(total_batch_number*1.0)

        #cost print
        print "Max Discriminator Cost : {}".format(Max_C.mean())
        print "Min Generator Cost : {}".format(Min_C.mean())

        #Save the data
        if epoch%save_period==0:
            print('Saving weights')
            modG.save_params("Weights/modG-{}.params" .format(epoch))
            modD_0.save_params("Weights/modD_0-{}.params"  .format(epoch))

    print "Optimization complete."

    #################################Generating Image####################################
    '''load method1 - load the training mod.get_params() directly'''
    #arg_params, aux_params = mod.get_params()

    '''load method2 - using the shared_module'''
    """
    Parameters
    shared_module : Module
        Default is `None`. This is used in bucketing. When not `None`, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
    """

    test_mod.bind(data_shapes=[mx.io.DataDesc(name='noise', shape=(column_size*row_size,noise_size))],label_shapes=None,shared_module=modG,for_training=False,grad_req='null')

    '''Annotate only when running test data. and Uncomment only if it is 'load method1' or 'load method2'''
    #test_mod.set_params(arg_params=arg_params, aux_params=aux_params)

    '''test_method-1'''
    '''
    noise = noise_iter.next()
    test_mod.forward(noise, is_train=False)
    result = test_mod.get_outputs()[0]
    result = result.asnumpy()
    print np.shape(result)
    '''

    '''test_method-2'''
    #'''
    test_mod.forward(data_batch=mx.io.DataBatch(data=[mx.random.normal(0, 1.0, shape=(column_size*row_size, noise_size))],label=None))
    result = test_mod.get_outputs()[0]
    print result
    result = result.asnumpy()

    #'''
    #visualization
    #fig ,  ax = plt.subplots(int(row_size/2.0), column_size, figsize=(column_size, int(row_size/2.0)))
    fig ,  ax = plt.subplots(row_size, column_size, figsize=(column_size, row_size))

    for i in xrange(column_size):
        #show 10 image

        #ax[i].set_axis_off()
        #ax[i].imshow(np.reshape(result[i],(28,28)))

        #show 20 image
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(result[i], (28, 28)),cmap='gray')
        ax[1][i].imshow(np.reshape(result[i+10], (28, 28)),cmap='gray')
    plt.show()
    #'''

if __name__ == "__main__":

    print "GAN_starting in main"
    GAN(epoch=100, noise_size=128, batch_size=128, save_period=100)

else:

    print "GAN_imported"
