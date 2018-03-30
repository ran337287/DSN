Pytorch implementation of DSNs([Domain Separation Networks](https://arxiv.org/pdf/1608.06019v1.pdf))

### Structure
We refer to the structure
![](https://github.com/ran337287/DSN/blob/master/image/structure.png?raw=true)


### Datasets
URL:

svhn https://pan.baidu.com/s/1Blp9l6sQdwqsfizmyvKVXA

mnist https://pan.baidu.com/s/1ap4srdVfZk-5s5ZM0OUnPw

    ln -s $DOWNLOAD_PATH $DSN/dataset/

### Train
python dsn_train.py

### Result
The accuracy on mnist test dataset can achieve 76.62%. The reconstructed images from left to right: reconstructed by private and shared feature;  reconstructed by shared feature; reconstructed by private feature.
![](https://github.com/ran337287/DSN/blob/master/image/mnist.png?raw=true)
![](https://github.com/ran337287/DSN/blob/master/image/svhn.png?raw=true)
