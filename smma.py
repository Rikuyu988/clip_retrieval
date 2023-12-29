import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import torchvision.transforms as transforms
import torch
from clip import CLIP

# 加载预训练的 CLIP 模型
clip = CLIP()

# 选择一些图片和相应的文本描述
image_paths = ['datasets\\flickr8k-images\\1397923690_d3bf1f799e.jpg',
               'datasets\\flickr8k-images\\2052202553_373dad145b.jpg',
               'datasets\\flickr8k-images\\549887636_0ea5ae4739.jpg',
               'datasets\\flickr8k-images\\1491192153_7c395991e5.jpg',
               "datasets\\flickr8k-images\\434938585_fbf913dfb4.jpg"]
text_descriptions = ['A person skis down a snowy hill .',
                     'A dog holds a stick in its mouth in the water .',
                     'A man standing in a crosswalk',
                     'A man putting a little boy wearing orange into a child swing .',
                     'a young girl standing outside a restaurant at night .']

device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载图像数据集
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])
# 编码图片和文本
image_features = []
text_features = []

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    image_input = data_transform(image).unsqueeze(0).to(device)
    image_feature = clip.net.encode_image(image_input)
    image_features.append(image_feature)

for text in text_descriptions:
    # text_input = clip.net.tokenize([text])
    text_feature = clip.net.encode_text(text)
    text_features.append(text_feature)

# 计算图片和文本之间的相似度
similarities = np.zeros((len(image_paths), len(text_descriptions)))

for i, image_feature in enumerate(image_features):
    for j, text_feature in enumerate(text_features):
        similarity = torch.nn.functional.cosine_similarity(image_feature, text_feature)
        similarities[i, j] = similarity.item()

# 创建大图
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制相似度矩阵
cax = ax.matshow(similarities, cmap='viridis')

# 显示图片
for i, image_path in enumerate(image_paths):
    image = io.imread(image_path)
    # ax.imshow(image, extent=(-0.5, len(text_descriptions) - 0.5, i + 0.5, i - 0.5), aspect='auto')
    ax.imshow(image, extent=(len(text_descriptions), len(text_descriptions) + 1, i + 0.5, i - 0.5), aspect='auto')
    # ax.imshow(image, extent=(len(text_descriptions) + 1, len(text_descriptions) + 1, i + 1, i - 1), aspect='auto')

# 在相似度矩阵格子中心显示相似度值
for i in range(len(image_paths)):
    for j in range(len(text_descriptions)):
        text = "%.2f" % similarities[i, j]  # 保留两位小数
        ax.text(j, i, text, ha='center', va='center', color='white', fontsize=8)

# 设置刻度标签和标题
ax.set_xticks(np.arange(len(text_descriptions))-0.5)
ax.set_xticklabels(text_descriptions, rotation=45, ha='left')
ax.set_yticks(np.arange(len(image_paths))-0.5)
ax.set_yticklabels([])
ax.set_xlabel('Text Descriptions')
ax.set_title('Similarity Matrix between Images and Text')

# 显示颜色条
fig.colorbar(cax)
plt.tight_layout()
plt.show()

