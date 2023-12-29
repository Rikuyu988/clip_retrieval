# clip_retrieval
AI Practice
基于clip模型改进的图像检索模型
本项目旨在在 CLIP 模型的基础上进行微调改进，提升图像检索的准确性和语义匹配能力，实现更精确、更语义化的图像搜索。不足较多。

## 功能特性

retrieval.py中输入文本描述，在指定数据集上查找对应五张图片

text_query = "A dog catching a Frisbee ."
![image](https://github.com/Rikuyu988/clip_retrieval/assets/130273480/013bd6b7-2e44-4389-a41d-0c1cf8270562)
text_query = "Beach with coconut trees"
![image](https://github.com/Rikuyu988/clip_retrieval/assets/130273480/68919a13-99e3-4714-b38d-1530dd05d7ec)

## 快速开始
1.准备数据集；
2.修改para.py中参数并运行开始微调，得到评估结果于logfile.log；
3.运行predict.py进行项目预测；

### 性能展示

不同参数模型评估指标对比
![image](https://github.com/Rikuyu988/clip_retrieval/assets/130273480/2be1e81a-fdb6-449d-9f83-e218c338ef56)

模型验证
![image](https://github.com/Rikuyu988/clip_retrieval/assets/130273480/17a262c3-2219-437e-9dfa-5e3fec18fbe7)

项目预测
![image](https://github.com/Rikuyu988/clip_retrieval/assets/130273480/682fab44-7b39-49b2-9794-1529f4f16dff)

