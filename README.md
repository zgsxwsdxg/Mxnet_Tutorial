
>## ***Mxnet?*** 
* ***Flexible and Efficient Library for Deep Learning***
* ***Symbolic programming or imperative programming***
* ***Mixed programming available*** *(Symbolic + imperative)*
 
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
    * [***Fully Connected Neural Network with LogisticRegressionOutput : Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Fully%20Connected%20Neural%20Network%20with_LogisticRegressionOutput)

    * [***Fully Connected Neural Network with SoftmaxOutput : Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Fully%20Connected%20Neural%20Network%20with_softmax)
    
    * [***Fully Connected Neural Network with SoftmaxOutput*** *(flexible)* ***: Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Fully%20Connected%20Neural%20Network%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Convolutional Neural Networks with SoftmaxOutput : Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Convolutional%20Neural%20Networks%20with%20SoftmaxOutput)

    * [***Convolutional Neural Networks with SoftmaxOutput*** *(flexible)* ***: Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Convolutional%20Neural%20Networks%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Recurrent Neural Networks with SoftmaxOutput : Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Recurrent%20Neural%20Networks%20with%20SoftmaxOutput)
    
    * [***Recurrent Neural Networks + LSTM with SoftmaxOutput : Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Recurrent%20Neural%20Networks%20%2B%20LSTM%20with%20SoftmaxOutput)

    * [***Recurrent Neural Networks + LSTM with SoftmaxOutput*** *(flexible)* ***: Classifying the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Recurrent%20Neural%20Networks%20%2B%20LSTM%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Autoencoder Neural Networks with logisticRegressionOutput : Compressing the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Autoencoder%20Neural%20Networks%20with%20logisticRegressionOutput)

    * [***Autoencoder Neural Networks with logisticRegressionOutput*** *(flexible)* ***: Compressing the MNIST data using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/basic/Autoencoder%20Neural%20Networks%20with%20logisticRegressionOutput(flexible%20to%20use%20the%20module))


* ### ***Neural Networks Applications***
    * [***Predicting lotto numbers in regression analysis using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/applications/Predicting%20lotto%20numbers%20in%20regression%20analysis%20using%20mxnet)

    * [***Generative Adversarial Networks with fullyConnected Neural Network : using the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/applications/Generative%20Adversarial%20Network%20with%20FullyConnected%20Neural%20Network)

    * [***Deep Convolution Generative Adversarial Network : using MNIST and ImageNet data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/applications/Deep%20Convolution%20Generative%20Adversarial%20Network)(**'Image net' is in progress**)

    * [***word2vec : using undefined data***]()(***not yet***)

>## ***Development environment***
* window 8.1 64bit 
* WinPython-64bit-2.7.10.3 - (Also available in Python package Library like Anaconda and so on)  
* pycharm Community Edition 2016.3.2 - (Also available in editors such as Spyder and Eclipse and so on.)

>## ***Dependencies*** 
* mxnet-0.9.4 or mxnet-0.9.5
* numpy-1.12.1
* matplotlib-1.5.0rc3
* opencv-3.2.0 (using **'import cv2'** in python)

