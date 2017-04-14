# -*- coding: utf-8 -*-
import mxnet as mx
import data_preprocessing as dp
from Network import LottoNet

'''implement'''

net=LottoNet(epoch=5,batch_size=10,save_period=1000)
print net[0]
print net[1]