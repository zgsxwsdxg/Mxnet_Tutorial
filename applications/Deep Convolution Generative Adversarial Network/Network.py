# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import cv2

'''unsupervised learning -Convolution Neural Netowrks  Generative Adversarial Networks'''

def to4d_tanh(img):

    '''1.resize (60000,64,64) -> and transform from 1 channel (60000,1,64,64) to 3 channel (60000,3,64,64)'''
    img = np.asarray([cv2.resize(i,(64,64),interpolation=cv2.INTER_CUBIC) for i in img ])
    img = img.reshape(img.shape[0], 1, 64, 64).astype(np.float32)
    img = np.tile(img, (1, 3, 1, 1)) # to 3channel

    #show image
    '''
    img = img.transpose((0, 2, 3, 1))
    cv2.imshow('orginal',img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    '''2. range conversion  0 ~ 255 -> -1 ~ 1 and '''
    img = (img/(255.0/2.0))-1.0

    return img

def Mnist_Data_Processing(batch_size):

    '''In this Gan tutorial, we don't need the label data.'''
    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz')

    '''train image + test image'''
    train_img = np.concatenate((train_img, test_img), axis=0)

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data': to4d_tanh(train_img)}, batch_size=batch_size, shuffle=True)  # training data
    return train_iter,len(train_img)

def Generator(relu ='relu',tanh='tanh',fix_gamma=True,eps=0.001,no_bias=True):

    '''
    Deep convolution Generative Adversarial Networks

    <Unique Point>
    1. no pooling, only with strided convolutions!!! -> okay
    2. Use Batch Normalization in both the generator and the discriminator,
    but not applying Batch Normalization to the generator output layer and the discriminator input layer -> okay
    3. Remove fully connected hidden layers for deeper architectures -> okay
    4. in generator, Use ReLU activation for all layers except for the output, which uses Tanh -> okay
    5. in discriminator, Use LeakyReLU activation in the discriminator for all layers, except for the output, which uses sigmoid -> okay

    <Details of Adversarial Training>
    1. noise data : uniform distribution -> okay
    2. No pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1] -> okay
    3. Using adam optimizer , learning rate = 0.0002 , B1 term is 0.5 -> okay
    4. mini-batch size 128 -> okay
    5. In the LeakyReLU, the slope of the leak was set to 0.2 in all models. -> okay

    <Networks Structure>
    cost_function - MIN_MAX cost_function
    '''
    #generator neural networks

    '''
    The first layer of the GAN, which takes a uniform noise distribution Z as input, could be called
    fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional
    tensor and used as the start of the convolution stack. For the discriminator, the last convolution layer
    is flattened and then fed into a single sigmoid output. See Fig. 1 for a visualization of an example
    model architecture.
    '''
    noise = mx.sym.Variable('noise') # The size of noise is 100.

    g1 = mx.sym.Deconvolution(noise, name='g1', kernel=(4,4), num_filter=512, no_bias=no_bias)
    gbn1 = mx.sym.BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type=relu)

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, no_bias=no_bias)
    gbn2 = mx.sym.BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type=relu)

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, no_bias=no_bias)
    gbn3 = mx.sym.BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type=relu)

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=64, no_bias=no_bias)
    gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type=relu)

    ### not applying Batch Normalization to the generator output layer ###
    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=3, no_bias=True)
    g_out = mx.sym.Activation(g5, name='g_out', act_type=tanh)

    return g_out

def Discriminator(leaky ='leaky',sigmoid='sigmoid',fix_gamma=True,eps=0.001,no_bias=True):

    #discriminator neural networks
    data = mx.sym.Variable('data') #(128,3,64,64)

    ### not applying Batch Normalization to the discriminator input layer ###
    d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=64, no_bias=no_bias) #weight ->  (num_filter, channel, kernel[0], kernel[1])
    dact1 = mx.sym.LeakyReLU(d1 , act_type=leaky, slope=0.2 , name='leaky1') #(128,64,32,32,)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, no_bias=no_bias) #weight ->  (num_filter, channel, kernel[0], kernel[1])
    dbn2 = mx.sym.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
    dact2 = mx.sym.LeakyReLU(dbn2 , act_type=leaky, slope=0.2 , name='leaky2') #(128,128,16,16)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, no_bias=no_bias) #weight ->  (num_filter, channel, kernel[0], kernel[1])
    dbn3 = mx.sym.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
    dact3 = mx.sym.LeakyReLU(dbn3 , act_type=leaky, slope=0.2 , name='leaky3') #(128,256,8,8)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=512, no_bias=no_bias) #weight ->  (num_filter, channel, kernel[0], kernel[1])
    dbn4 = mx.sym.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
    dact4 = mx.sym.LeakyReLU(dbn4 , act_type=leaky, slope=0.2 , name='leaky4') #(128,512,4,4)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=True) #(128,1,1,1)

    '''For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output. '''
    d_out = mx.sym.Flatten(d5)
    d_out=mx.sym.Activation(data=d_out,act_type=sigmoid,name="d_out")

    '''expression-1'''
    out1 = mx.sym.MakeLoss(mx.symbol.log(d_out),grad_scale=-1.0,normalization='batch',name="loss1")
    out2 = mx.sym.MakeLoss(mx.symbol.log(1.0-d_out),grad_scale=-1.0,normalization='batch',name='loss2')

    '''expression-2,
    question? Why multiply the loss equation by -1?
    answer : for Maximizing the Loss function , and This is because mxnet only provides optimization techniques that minimize.
    '''
    '''
    Why two 'losses'?
    If you put the label variable in the network, you can configure the loss to be one, but the network is not learning well. I do not know why.
    '''
    #out1 = mx.sym.MakeLoss(-1.0*mx.symbol.log(d_out),grad_scale=1.0,normalization='batch',name="loss1")
    #out2 = mx.sym.MakeLoss(-1.0*mx.symbol.log(1.0-d_out),grad_scale=1.0,normalization='batch',name='loss2')

    group=mx.sym.Group([out1,out2])

    return group

def DCGAN(epoch,noise_size,batch_size,save_period,dataset):


    if dataset == "MNIST":
        train_iter,train_data_number = Mnist_Data_Processing(batch_size)
        #No need, but must be declared.
        label =mx.nd.zeros((batch_size,))
    elif dataset == 'ImageNet':
        pass

    '''Network'''
    generator=Generator()
    discriminator=Discriminator()

    '''In the code below, the 'inputs_need_grad' parameter in the 'mod.bind' function is very important.'''

    # =============module G=============
    modG = mx.mod.Module(symbol=generator, data_names=['noise'], label_names=None, context= mx.gpu(0))
    modG.bind(data_shapes=[('noise', (batch_size, noise_size,1,1))],label_shapes=None,for_training=True)

    #load the saved modG data
    #modG.load_params("Weights/modG-100.params")

    modG.init_params(initializer=mx.initializer.Normal(sigma=0.02))
    modG.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.0002,'beta1' : 0.5})


    # =============module discriminator[0],discriminator[1]=============
    modD_0 = mx.mod.Module(symbol=discriminator[0], data_names=['data'], label_names=None, context= mx.gpu(0))
    modD_0.bind(data_shapes=train_iter.provide_data,label_shapes=None,for_training=True,inputs_need_grad=True)

    # load the saved modD_O data
    #modD_0.load_params("Weights/modD_0-100.params")

    modD_0.init_params(initializer=mx.initializer.Normal(sigma=0.02))
    modD_0.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.0002,'beta1' : 0.5})

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
    for epoch in xrange(epoch):
        Max_cost_0=0
        Max_cost_1=0
        Min_cost=0
        total_batch_number = np.ceil(train_data_number/(batch_size*1.0))
        train_iter.reset()
        for batch in train_iter:
            ################################updating only parameters related to modD.########################################
            # updating discriminator on real data
            '''MAX : modD_0 : -mx.symbol.log(discriminator2)  real data Discriminator update , bigger and bigger discriminator2'''
            modD_0.forward(data_batch=batch, is_train=True)
            modD_0.backward()
            modD_0.update()

            '''Max_Cost of real data Discriminator'''
            Max_cost_0-=modD_0.get_outputs()[0].asnumpy()

            # update discriminator on noise data
            '''MAX : modD_1 :-mx.symbol.log(1-discriminator2)  - noise data Discriminator update , bigger and bigger -> smaller and smaller discriminator2'''
            noise = mx.random.uniform(low=0.0, high=1.0, shape=(batch_size, noise_size, 1, 1), ctx=mx.gpu(0))
            modG.forward(data_batch=mx.io.DataBatch(data=[noise],label=None), is_train=True)
            modG_output = modG.get_outputs()

            modD_1.forward(data_batch=mx.io.DataBatch(data=modG_output,label=None), is_train=True)
            modD_1.backward()
            modD_1.update()

            '''Max_Cost of noise data Discriminator'''
            Max_cost_1-=modD_0.get_outputs()[0].asnumpy()

            ################################updating only parameters related to modG.########################################
            # update generator on noise data
            '''MIN : modD_0 : -mx.symbol.log(discriminator2) - noise data Discriminator update  , bigger and bigger discriminator2'''
            modD_0.forward(data_batch=mx.io.DataBatch(data=modG_output, label=None), is_train=True)
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

        print "epoch : {}".format(epoch+1)

        #Save the data
        if (epoch+1)%save_period==0:
            print('Saving weights')
            modG.save_params("Weights/modG-{}.params" .format(epoch+1))
            modD_0.save_params("Weights/modD_0-{}.params"  .format(epoch+1))

    print "Optimization complete."

    #################################Generating Image####################################
    '''load method1 - load the training mod.get_params() directly'''
    #arg_params, aux_params = mod.get_params()

    '''Annotate only when running test data. and Uncomment only if it is 'load method2' '''
    #test_mod.set_params(arg_params=arg_params, aux_params=aux_params)

    '''test_method-1'''
    '''
    noise = noise_iter.next()
    test_mod.forward(noise, is_train=False)
    result = test_mod.get_outputs()[0]
    result = result.asnumpy()
    print np.shape(result)
    '''
    '''load method2 - using the shared_module'''
    """
    Parameters
    shared_module : Module
        Default is `None`. This is used in bucketing. When not `None`, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
    """

    column_size=10; row_size=10
    test_mod.bind(data_shapes=[mx.io.DataDesc(name='noise', shape=(column_size*row_size,noise_size,1,1))],label_shapes=None,shared_module=modG,for_training=False,grad_req='null')

    '''test_method-2'''
    test = mx.random.uniform(low=0.0, high=1.0, shape=(batch_size, noise_size, 1, 1), ctx=mx.gpu(0))
    test_mod.forward(data_batch=mx.io.DataBatch(data=[test],label=None))
    result = test_mod.get_outputs()[0]
    result = result.asnumpy()

    '''range adjustment  -1 ~ 1 -> 0 ~ 1 -> 0 ~ 255 '''
    result = ((result+1.0)/2.0)*255.0

    result = result.transpose((0, 2, 3, 1))

    '''visualization'''
    fig ,  ax = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
    fig.suptitle('generator')
    for j in xrange(row_size):
        for i in xrange(column_size):
            ax[j][i].set_axis_off()
            if dataset == "MNIST":
                ax[j][i].imshow(result[i+j*column_size],cmap='gray')
            elif dataset == 'ImageNet':
                ax[j][i].imshow(result[i+j*column_size])
    plt.show()

    if dataset == "MNIST":
        plt.savefig("DCGAN_MNIST.png")
    elif dataset == 'ImageNet':
        plt.savefig("DCGAN_ImageNet.png")

if __name__ == "__main__":
    print "GAN_starting in main"
    DCGAN(epoch=5, noise_size=100, batch_size=128, save_period=100,dataset='MNIST')
else:
    print "GAN_imported"
