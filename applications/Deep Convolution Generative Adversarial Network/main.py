import Network
import argparse
'''I wrote this code with reference to  https://github.com/dmlc/mxnet/blob/master/example/gan/dcgan.py.
    I tried to make it easy to understand.
'''
'''
I initialized the Hyperparameters values introduced in 'DETAILS OF ADVERSORIAL TRAINING part'
of 'UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS' paper.
'''
FLAGS=True
parser=argparse.ArgumentParser(description='hyperparameters')
parser.add_argument("--epoch",action="store_const",type=int)
parser.add_argument("--noise_size",action="store_const",type=int)
parser.add_argument("--batch_size",action="store_const",type=int)
parser.add_argument("--save_period",action="store_const",type=int)
parser.add_argument("--dataset",action="store",type=str)
parser.add_argument("--state",action="store_true",dest="state",type=False)
args = parser.parse_args()

if FLAGS:
    Network.DCGAN(epoch=30, noise_size=100, batch_size=200, save_period=30,dataset='CIFAR10')
elif args.state:
    Network.DCGAN(epoch=args.epoch, noise_size=args.noise_size, batch_size=args.batch_size, save_period=args.save_period, dataset=args.dataset)