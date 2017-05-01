# -*- coding: utf-8 -*-
import mxnet as mx
import data_preprocessing as dp
from Network import LottoNet

'''implement'''
net=LottoNet(epoch=100,batch_size=100,save_period=100)
print net[0]
#print net[1]