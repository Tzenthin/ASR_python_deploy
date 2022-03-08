# ASR_python_deploy
本项目是基于python，对语音识别服务进行的部署。实验使用的ASR模型是wenet的开源模型，实际上，任何一个支持一句话解码的ASR模型，都可参考本框架部署自己的语音识别服务。
具体的文字介绍，可参考知乎：https://zhuanlan.zhihu.com/p/467364921 上的文章。

部署的方式为自己设计，解码模型可直接使用wenet的预训练模型（该预训练模型是基于wenet speech数据集训练而成），可参考一下链接。
https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/20211025_conformer_exp.tar.gz
解压后的4个文件，final.pt  global_cmvn  train.yaml  words.txt，放至conf/ 目录下即可。

其他的关于运行环境，就是wenet的训练环境，模型的训练和测试，可移步到进行学习：
https://github.com/wenet-e2e/wenet
