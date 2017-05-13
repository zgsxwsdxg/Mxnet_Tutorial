
>## ***Mxnet?*** 
* ***Flexible and Efficient Library for Deep Learning***
* ***Symbolic programming or imperative programming***
* ***Mixed programming available*** ***(`Symbolic + imperative`)***
 
<image src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png" width=800 height=200></image>
>## ***Introduction*** 
*   
    It is a tutorial that can be helpful to those `who are new to the MXNET deep-learning framework`
>## ***Official Homepage Tutorial***
*
    The following LINK is a tutorial on the MXNET  official homepage
    * `Link` : [mxnet homapage tutorials](http://mxnet.io/tutorials/index.html)
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

* ### ***Neural Networks basic with visualization***
    * [***mxnet with graphviz library***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/visualization/visualization)
        ```cmd
        <linux>
        pip install graphviz(in anaconda Command Prompt) 

        <window>
        1. 'pip install graphviz'(in winpython Command Prompt) 
        2. download 'graphviz-2.38.msi' at 'http://www.graphviz.org/Download_windows.php'
        3. Install to 'C:\Program Files (x86)\'
        4. add 'C:\Program Files (x86)\Graphviz2.38\bin' to the environment variable PATH
        ```
        ```python
        Must be run on 'jupyter notebook'   

        import mxnet as mx  
        ...
        ...
        mx.viz.plot_network(symbol=mlp, shape=shape)
        ```
        * `To view only the results, run the 'xxx.html' file with 'Chrome' or 'Explorer'`
    * [***mxnet with tensorboard Only available on Linux***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/visualization/tensorboard-linux)
        
        ```python
        pip install tensorboard   
        ```
        ```python
        '''Issue'''
        The '80's line of the tensorboard file in the path '/home/user/anaconda2/bin' should be modified as shown below.
        ```
        ```python
        <code>
        for mod in package_path:
            module_space = mod + '/tensorboard/tensorboard' + '.runfiles'
            if os.path.isdir(module_space):
                return module_space
        ```
        * If you want to see the results immediately,`write the following script in the terminal window` where the event file exists.
        
            * `tensorboard --logdir=tensorboard --logdir=./ --port=6006`


* ### ***Neural Networks Applications***
    * [***Predicting lotto numbers in regression analysis using mxnet***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/applications/Predicting%20lotto%20numbers%20in%20regression%20analysis%20using%20mxnet)

    * [***Generative Adversarial Networks with fullyConnected Neural Network : using MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/applications/Generative%20Adversarial%20Network%20with%20FullyConnected%20Neural%20Network)

    * [***Deep Convolution Generative Adversarial Network : using ImageNet , CIFAR10 , MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/applications/Deep%20Convolution%20Generative%20Adversarial%20Network)
       ```cmd
        <Code execution example>  
        python main.py --state --epoch 100 --noise_size 100 --batch_size 200 --save_period 100 --dataset CIFAR10```
>## ***Development environment***
* ```window 8.1 64bit``` and ```Ubuntu linux 16.04.2 LTS``` 
* `WinPython-64bit-2.7.10.3(window) and anaconda2 4.3.1(linux)` 
* `pycharm Community Edition 2016.3.2` - (Also available in editors such as Spyder and Eclipse and so on.)

>## ***Dependencies*** 
* mxnet-0.9.5
* numpy-1.12.1, matplotlib-2.0.1 , tensorboard-1.0.0a7(linux) , graphviz -> (`Visualization`)
* opencv-3.2.0(window) , opencv-3.1.0(linux) , struct , gzip , os , glob , threading -> (`Data preprocessing`)
* cPickle -> (`Data save and restore`)
* logging -> (`Observation during learning`)
* argparse -> (`Command line input from user`)
* urllib , requests -> (`Web crawling`) 
