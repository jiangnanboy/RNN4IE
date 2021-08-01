# RNN4IE

中文信息抽取工具。使用RNN的不同结构进行信息抽取，该项目使用pytorch，python开发。

**Guide**

- [Intro](#Intro)
- [Model](#Model)
- [Evaluate](#Evaluate)
- [Install](#install)
- [Dataset](#Dataset)
- [Todo](#Todo)
- [Cite](#Cite)
- [Reference](#reference)

## Intro

目前主要实现中文实体抽取：

训练样本以B、I、O形式进行标注。

## Model
### 模型
* gru(rnn4ie/gru)：此模型利用【gru + crf】进行实体抽取。

![image](https://raw.githubusercontent.com/jiangnanboy/RNN4IE/master/rnn4ie/gru/model.png)

* gru_mhsa(rnn4ie/gru_mhsa)：此模型利用【gru + multi-head-self-attention + crf】进行实体抽取。

![image](https://raw.githubusercontent.com/jiangnanboy/RNN4IE/master/rnn4ie/gru_mhsa/model.png)

* gru_sa(rnn4ie/gru_sa)：此模型利用【gru + soft-attention + crf】进行实体抽取。

![image](https://raw.githubusercontent.com/jiangnanboy/RNN4IE/master/rnn4ie/gru_sa/model.png)

* gru_xca(rnn4ie/gru_xca)：此模型利用【gru + cross-covariance-attention + crf】进行实体抽取。

![image](https://raw.githubusercontent.com/jiangnanboy/RNN4IE/master/rnn4ie/gru_xca/model.png)

#### Usage
- 配置文件

    各个model在训练和预测时需加载不同的配置文件config.cfg，各个model的config.cfg内容见：
    * [gru_cfg](rnn4ie/gru/config.cfg)
    * [gru_mhsa_cfg](rnn4ie/gru_mhsa/config.cfg)
    * [gru_sa_cfg](rnn4ie/gru_sa/config.cfg)
    * [gru_xca_cfg](rnn4ie/gru_xca/config.cfg)
    
    

- 训练(支持加载预训练的embedding向量)
    ```
    from rnn4ie.gru.train import Train
  
    train = Train()
    train.train_model('config.cfg')
  ---------------------------------
    from rnn4ie.gru_mhsa.train import Train
  
    train = Train()
    train.train_model('config.cfg')
  ---------------------------------
    from rnn4ie.gru_sa.train import Train
  
    train = Train()
    train.train_model('config.cfg')
  ---------------------------------
    from rnn4ie.gru_xca.train import Train
  
    train = Train()
    train.train_model('config.cfg')
    ```
      
- 预测

    ```
    from rnn4ie.gru.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config_cfg')
    result = predict.predict('据新华社报道，安徽省六安市被评上十大易居城市！')
  ---------------------------------
    from rnn4ie.gru_mhsa.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config_cfg')
    result = predict.predict('据新华社报道，安徽省六安市被评上十大易居城市！')
  ---------------------------------
    from rnn4ie.gru_sa.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config_cfg')
    result = predict.predict('据新华社报道，安徽省六安市被评上十大易居城市！')
  ---------------------------------
    from rnn4ie.gru_xca.predict import Predict
  
    predict = Predict()
    predict.load_model_vocab('config_cfg')
    result = predict.predict('据新华社报道，安徽省六安市被评上十大易居城市！')
    ```
## Evaluate
评估采用的是P、R、F1、PPL等。评估方法可利用scikit-learn中的precision_recall_fscore_support或classification_report。


## Install
* 安装：pip install RNN4IE
* 下载源码：
```
git clone https://github.com/jiangnanboy/RNN4IE.git
cd RNN4IE
python setup.py install
```


通过以上两种方法的任何一种完成安装都可以。如果不想安装，可以下载[github源码包](https://github.com/jiangnanboy/RNN4IE/archive/master.zip)

## Dataset

   这里利用data(来自人民日报，识别的是[ORG, PER, LOC, T, O])中的数据进行训练评估。
    
   预训练embedding向量：[sgns.sogou.char.bz2](https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw)

数据集的格式见[data](data/)，分为train与dev，其中source与target为中文对应的实体标注。

数据被处理成csv格式。

## Todo
持续加入更多模型......

## Cite

如果你在研究中使用了RNN4IE，请按如下格式引用：

```latex
@software{RNN4IE,
  author = {Shi Yan},
  title = {RNN4IE: Chinese Information Extraction Tool},
  year = {2021},
  url = {https://github.com/jiangnanboy/RNN4IE},
}
```

## License

RNN4IE 的授权协议为 **Apache License 2.0**，可免费用做商业用途。请在产品说明中附加RNN4IE的链接和授权协议。RNN4IE受版权法保护，侵权必究。

## Reference

* [Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)
* [facebookresearch](https://github.com/facebookresearch/xcit)
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)