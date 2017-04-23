import Network

'''I wrote this code with reference to  https://github.com/dmlc/mxnet/blob/master/example/gan/dcgan.py.
    I tried to make it easy to understand.
'''
'''
I initialized the Hyperparameters values introduced in 'DETAILS OF ADVERSORIAL TRAINING part'
of 'UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS' paper.
'''
Network.DCGAN(epoch=10, noise_size=100, batch_size=128, save_period=100,dataset='MNIST')