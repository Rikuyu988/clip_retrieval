import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from clip import CLIP


class ImageRetrievalModel(object):
    def __init__(self, clip_model, img_features):
        self.clip_model = clip_model
        self.img_features = img_features  # 图片特征数据

    def image_retrieval(self, text_query):
        # 对文本查询进行编码，获取文本特征表示
        with torch.no_grad():
            # text_input = self.model.net.tokenize([text_query]).to('cuda' if torch.cuda.is_available() else 'cpu')
            text_features = self.clip_model.net.encode_text(text_query).to('cuda' if torch.cuda.is_available() else 'cpu')

        # 计算文本与图片之间的相似度
        similarities = torch.nn.functional.cosine_similarity(text_features, self.img_features).cpu().numpy()

        # 获取相似度最高的五张图片索引和对应相似度得分
        top_indices = np.argsort(similarities)[::-1][:5]
        top_similar_images = [(similarities[idx], idx) for idx in top_indices]

        return top_similar_images

# 加载CLIP模型
clip = CLIP()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义文本查询
text_query = "Beach with coconut trees"

# 加载图像数据集
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.RandomRotation(15),  # 随机旋转
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩变换
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])
image_dataset = ImageFolder(root=r'E:\pythonProject\Corel5k', transform=data_transform)
image_features = []

# 提取图片特征
with torch.no_grad():
    for path, _ in image_dataset.samples:
        image = Image.open(path).convert("RGB")
        image_input = data_transform(image).unsqueeze(0).to(device)
        image_feature = clip.net.encode_image(image_input)
        image_features.append(image_feature)

    image_features = torch.cat(image_features, dim=0)  # 将特征拼接成一个张量

# 创建图像检索模型
retrieval_model = ImageRetrievalModel(clip_model=clip, img_features=image_features)

# 执行图像检索
similar_images = retrieval_model.image_retrieval(text_query)

# 显示相似的五个图像和对应的相似度得分
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6))
for i, (similarity_score, idx) in enumerate(similar_images):
    formatted_score = "{:.5f}".format(similarity_score)
    image_path = image_dataset.samples[idx][0]

    # 显示图片和图片名称
    plt.subplot(1, 5, i + 1)
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.title(f"Similarity: {formatted_score}")
    plt.axis('off')  # 不显示坐标轴

plt.tight_layout()
plt.show()
