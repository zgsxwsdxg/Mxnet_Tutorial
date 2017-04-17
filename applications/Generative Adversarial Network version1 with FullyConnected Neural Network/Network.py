# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

'''unsupervised learning -  Autoencoder'''


def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255

def get_noise(batch_size,noise_data):
    return mx.nd.normal(loc=0,scale=1,shape=(batch_size,noise_data))

def data_processing(batch_size):
    '''In this Gan tutorial, we don't need the label data.'''
    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    #train_iter  = mx.io.NDArrayIter(data={'data' : to2d(train_img)},label={'label' : train_lbl_one_hot}, batch_size=batch_size, shuffle=True) #training data
    train_iter = mx.io.NDArrayIter(data={'data': to2d(train_img)}, batch_size=batch_size, shuffle=True)  # training data
    noise_iter = mx.io.NDArrayIter(data={'noise': get_noise(batch_size,128)}, batch_size=batch_size)  #training data#noise data

    return train_iter,noise_iter

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
    discriminator2 = mx.sym.Activation(data=d_affine2, name='d_sigmoid2', act_type='sigmoid')
    d_out=mx.sym.LogisticRegressionOutput(data=discriminator2,label=label,name='d_loss')
    return d_out

def GAN(epoch,batch_size,save_period):

    train_iter, noise_iter = data_processing(batch_size)
    label = mx.nd.zeros(shape=(batch_size,))
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

    #generator mod
    g_mod = mx.mod.Module(symbol=generator, data_names=['noise'],label_names=None, context=mx.gpu(0))
    g_mod.bind(data_shapes=noise_iter.provide_data)
    g_mod.init_params(initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=1),)
    g_mod.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001})

    #discriminator mod
    d_mod = mx.mod.Module(symbol=discriminator, data_names=['data'],label_names=['label'], context=mx.gpu(0))
    d_mod.bind(data_shapes=train_iter.provide_data)
    d_mod.init_params(initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=1),)
    d_mod.init_optimizer(optimizer='adam',optimizer_params={'learning_rate': 0.001})

    #In mxnet,I think Implementing the Gan code is harder to implement than anything framework.
    ####################################training loop############################################
    for batch in noise_iter:
        g_mod.forward(batch, is_train=True)  # compute predictions
        g_mod.get_outputs()
        print g_mod.get_outputs()[0]





    #################################Generating Image####################################
    '''all data test'''
    result = g_mod.predict(noise_iter).asnumpy()

    '''visualization'''
    print_size=10
    fig ,  ax = plt.subplots(1, print_size, figsize=(print_size, 1))
    #fig ,  ax = plt.subplots(2, print_size, figsize=(print_size, 2))

    for i in xrange(print_size):
        '''show 10 image'''
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(result[i],(28,28)))
        '''
        # show 20 image
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(result[i], (28, 28)))
        ax[1][i].imshow(np.reshape(result[i+10], (28, 28)))
        '''
    plt.show()

if __name__ == "__main__":

    print "NeuralNet_starting in main"
    GAN(epoch=100,batch_size=100,save_period=10)

else:

    print "NeuralNet_imported"
