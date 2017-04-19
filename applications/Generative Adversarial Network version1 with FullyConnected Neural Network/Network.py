# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

'''unsupervised learning - Generative Adversarial Network'''

def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255

''' To use the module function. We must write the following code that inherits the dataiter. '''
class NoiseIter(mx.io.DataIter):

    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('noise', (batch_size, ndim))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim))]

def Data_Processing(batch_size):
    '''In this Gan tutorial, we don't need the label data.'''
    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data': to2d(train_img)}, batch_size=batch_size, shuffle=True)  # training data

    return train_iter

def Generator():
    #generator neural networks
    noise = mx.sym.Variable('noise') # The size of noise is 128.
    g_affine1 = mx.sym.FullyConnected(data=noise,name='g_affine1',num_hidden=256)
    generator1 = mx.sym.Activation(data=g_affine1, name='g_sigmoid1', act_type='sigmoid')
    g_affine2 = mx.sym.FullyConnected(data=generator1, name='g_affine2', num_hidden=784)
    g_out= mx.sym.Activation(data=g_affine2, name='g_sigmoid2', act_type='sigmoid')
    return g_out

def Discriminator():
    #discriminator neural networks
    data = mx.sym.Variable('data') # The size of data is 784(28*28)
    label = mx.sym.Variable('label')
    d_affine1 = mx.sym.FullyConnected(data=data,name='d_affine1',num_hidden=256)
    discriminator1 = mx.sym.Activation(data=d_affine1,name='d_sigmoid1',act_type='sigmoid')
    d_affine2 = mx.sym.FullyConnected(data=discriminator1,name = 'd_affine2' , num_hidden=1)
    discriminator2= mx.sym.Activation(data=d_affine2, name='d_sigmoid2', act_type='sigmoid')
    d_out=mx.sym.LinearRegressionOutput(data=discriminator2 ,label=label,name='d_loss')
    #d_out = mx.sym.LogisticRegressionOutput(data=d_affine2 , label=label, name='d_loss')
    return d_out

def GAN(epoch,batch_size,save_period):

    save_weights=False
    save_path="Weights/"

    train_iter = Data_Processing(batch_size)
    noise_iter = NoiseIter(batch_size, 128)

    label = mx.nd.zeros((batch_size))
    print np.shape(label)

    '''
    Generative Adversarial Networks

    <structure>
    generator - 128 - 256 - (784 image generate)

    discriminator -  784 - 256 - (1 Identifies whether the image is an actual image or not)

    cost_function - MIN_MAX cost_function
    '''
    '''Network'''
    generator=Generator()
    discriminator=Discriminator()

    # =============module G=============
    modG = mx.mod.Module(symbol=generator, data_names=['noise'], label_names=None, context= mx.gpu(0))
    modG.bind(data_shapes=noise_iter.provide_data)

    # load the modG weights
    #modG.load_params(save_path+"modG-100.params")

    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001})

    # =============module D=============
    modD = mx.mod.Module(symbol=discriminator, data_names=['data'], label_names=['label'], context= mx.gpu(0))
    modD.bind(data_shapes=train_iter.provide_data,label_shapes=[('label', (batch_size,))],inputs_need_grad=True)
    '''
    !!!caution!!!!
    inputs_need_grad (bool) – Default is False. Whether the gradients to the input data need to be computed.
    Typically this is not needed. But this might be needed when implementing composition of modules.
    '''
    #load the modD weights
    #modD.load_params(save_path+"modD-100.params")

    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001})

    # =============generate image=============
    test_mod = mx.mod.Module(symbol=generator, data_names=['noise'], label_names=None, context= mx.gpu(0))

    #In mxnet,I think Implementing the Gan code is harder to implement than anything framework.
    ####################################training loop############################################

    '''
    class mxnet.metric.CustomMetric(feval, name=None, allow_extra_outputs=False)
    Custom evaluation metric that takes a NDArray function.
    '''
    def cost_Function(label, pred):
        pred = pred.ravel()
        #return -(np.log(pred + 1e-12) + np.log(1. - pred + 1e-12)).mean()
        label = label.ravel()
        return -(label * np.log(pred + 1e-12) + (1. - label) * np.log(1. - pred + 1e-12)).mean()


    #loss_generator
    mG = mx.metric.CustomMetric(cost_Function)

    #loss_discriminator
    mD = mx.metric.CustomMetric(cost_Function)

    # =============training===============
    for epoch in xrange(1,epoch+1,1):
        print "epoch : {}".format(epoch)
        train_iter.reset()
        for batch in train_iter:

            '''noize->generator'''
            noise = noise_iter.next()
            modG.forward(noise, is_train=True)
            outG = modG.get_outputs()
            ########################################## update discriminator on fake##############################################
            '''generator -> discriminator'''
            label[:]= 0
            #print [label]
            modD.forward(mx.io.DataBatch(outG,[label]), is_train=True)
            modD.backward()

            '''
            def backward(self, out_grads=None):
            Backward computation.

            Parameters
             ----------
            out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.

            '''
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

            #update discriminator on fake, only updating discriminator variables
            #modD.update_metric(mD, [label])

            ########################################## update discriminator on real##############################################
            label[:] = 1
            batch.label = [label]
            modD.forward(batch, is_train=True)
            modD.backward()
            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            modD.update()

            # update discriminator on real, only updating discriminator variables
            #modD.update_metric(mD, [label])

            ##########################################update generator##############################################
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD.get_input_grads()

            '''
            one more time reading!!
            def backward(self, out_grads=None):
            Backward computation.

            Parameters
             ----------
            out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.

            '''
            modG.backward(diffD)
            modG.update()

            # update generator, only updating generator variables
            mG.update([label], modD.get_outputs())

        #Save the data
        if save_weights and epoch%save_period==0:
            print('Saving weights')
            modG.save_params(save_path+"modG-{}.params" .format(epoch))
            modD.save_params(save_path+"modD-{}.params"  .format(epoch))


    #################################Generating Image####################################
    arg_params, aux_params=modG.get_params() #load 'Learning completed parameters'
    test_mod.bind(data_shapes=noise_iter.provide_data,label_shapes=None, for_training=False)
    test_mod.set_params(arg_params=arg_params, aux_params=aux_params)

    '''generating image'''
    noise = noise_iter.next()
    test_mod.forward(noise,is_train=False)
    result = test_mod.get_outputs()[0]
    result=result.asnumpy()
    print result
    '''visualization'''

    column_size=10
    #fig ,  ax = plt.subplots(1, print_size, figsize=(print_size, 1))
    fig ,  ax = plt.subplots(2, column_size, figsize=(column_size, 2))

    for i in xrange(column_size):
        '''show 10 image'''
        '''
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(result[i],(28,28)))
        '''
        #'''
        '''show 20 image'''
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(result[i], (28, 28)))
        ax[1][i].imshow(np.reshape(result[i+10], (28, 28)))
        #'''
    plt.show()

if __name__ == "__main__":

    print "NeuralNet_starting in main"
    GAN(epoch=100,batch_size=100,save_period=100)

else:

    print "NeuralNet_imported"
