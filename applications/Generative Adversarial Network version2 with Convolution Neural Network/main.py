import Network

'''I wrote this code with reference to  https://github.com/dmlc/mxnet/blob/master/example/gan/dcgan.py.'''
Network.GAN(epoch=50,noise_size=100,batch_size=100,save_period=100)