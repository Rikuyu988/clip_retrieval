# clip_retrieval
AI Practice
基于clip模型改进的图像检索模型
本项目旨在在 CLIP 模型的基础上进行微调改进，提升图像检索的准确性和语义匹配能力，实现更精确、更语义化的图像搜索。不足较多。

## 功能特性

retrieval.py中输入文本描述，在指定数据集上查找对应五张图片

text_query = "A dog catching a Frisbee ."
![image](img\dog.png)
text_query = "Beach with coconut trees"
![image](img\beach.png)

## 快速开始
1.准备数据集；
2.修改para.py中参数并运行开始微调，得到评估结果于logfile.log；
3.运行predict.py进行项目预测；

### 性能展示

不同参数模型评估指标对比
![image](img\model.png)

模型验证
![image](img\predict.png)


