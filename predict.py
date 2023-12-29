from PIL import Image
import matplotlib.pyplot as plt
from clip1 import CLIP

if __name__ == "__main__":
    clip = CLIP()

    # 图片的路径
    image_path = "img/2090545563_a4e66ec76b.jpg"
    # 寻找对应的文本，4选1
    captions = [
        "The two children glided happily on the skateboard.",
        "A woman walks through a barrier while everyone else is backstage.",
        "A white dog was watching a black dog jump on the grass next to a pile of big stones.",
        "An outdoor skating rink was crowded with people."
    ]

    image = Image.open(image_path)
    probs = clip.detect_image(image, captions)
    print("Label probs:", probs)

    # 显示图片
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.axis('off')

    # 定义文本框的位置和大小
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.1)

    # 循环显示每个文本描述和对应的预测值
    for i, caption in enumerate(captions):
        text = f"{caption}\nProbability: {probs[0][i]:.4f}"
        ax.annotate(text, xy=(0, 0), xytext=(0.1, 0.9 - i * 0.1), textcoords="axes fraction",
                    fontsize=8, ha='left', va='top', bbox=bbox_props)

    plt.show()
