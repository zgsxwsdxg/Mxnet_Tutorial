import Network

'''This code is based on https://github.com/dmlc/mxnet/blob/master/example/gan/dcgan.py.'''

Network.GAN(epoch=10,noise_size=128,batch_size=100,save_period=100)