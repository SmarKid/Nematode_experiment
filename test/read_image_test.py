from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img_path_name = 'E:\workspace\线虫数据集\图片整理\\6_0.417_0.500\\b_1_2_6_2.JPG'
    img_PIL = Image.open(img_path_name)
    plt.imshow(img_PIL)
    plt.show()