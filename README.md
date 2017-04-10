
>## ***Mxnet?*** 
***Flexible and Efficient Library for Deep Learning***

<image src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png" width=800 height=200></image>
>## ***Introduction*** 
*   
    It is a tutorial that can be helpful to those __who are new to the MXNET deep-learning framework__
>## ***Official Homepage Tutorial***
*
    The following LINK is a tutorial on the MXNET  official homepage

* Link : [mxnet homapage tutorials](http://mxnet.io/tutorials/index.html)
>## ***Let's begin with***
* Required library and very simple code
```python
import mxnet as mx
import numpy as np

out=mx.nd.ones((3,3),mx.gpu(0))
print mx.asnumpy(out)
```
* The below code is the result of executing the above code
```
<NDArray 3x3 @gpu(0)>
[[ 1.  1.  1.]
 [ 1.  1.  1.]
 [ 1.  1.  1.]]
```        
>## ***Topics***
* ### ***Neural Networks basic***
    * [***Fully Connected Neural Network with LogisticRegressionOutput : Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Mnist_FullyNeuralNetwork_mxnet%20with_LogisticRegressionOutput)
    * [***Fully Connected Neural Network with SoftmaxOutput : Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Mnist_FullyNeuralNetwork_mxnet%20with_softmax)
    * [***Convolutional Neural Networks with SoftmaxOutput : Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Convolutional%20Neural%20Networks%20with%20SoftmaxOutput)
    * [***Recurrent Neural Networks with SoftmaxOutput : Classifying the MNIST data using mxnet***]()(***not yet***)
    * [***Recurrent Neural Networks with LSTM with SoftmaxOutput: Classifying the MNIST data using mxnet***]()(***not yet***)
    * [***Autoencoder Neural Networks with logisticRegressionOutput : Compressing the MNIST data using mxnet***]()(***not yet***)
* ### ***Neural Networks Applications***
    * [***Predicting lotto numbers in regression analysis using mxnet version1***]()
    * [***Predicting lotto numbers in regression analysis using mxnet version2***]()
    * [***Generative Adversarial Network : using undefined data***]()(***not yet***)
    * [***word2vec : using undefined data***]()(***not yet***)

>## ***Development environment***
* window 8.1 64bit 
* WinPython-64bit-2.7.10.3 - (Also available in Python package Library like Anaconda and so on)  
* mxnet-0.9.4
* pycharm Community Edition 2016.3.2 - (Also available in editors such as Spyder and Eclipse and so on.)
>## ***Dependencies*** 
+ mxnet-0.9.4
+ numpy-1.12.0

