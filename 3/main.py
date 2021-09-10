import tensorflow as tf
import random
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from hamming import HammingNetwork


def getNoisyBinaryImage(factor, path, save_path):
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    t = []
    for i in range(width):
        for j in range(height):
            rand = random.randint(-factor, factor)
            a = pix[i, j][0] + rand
            b = pix[i, j][1] + rand
            c = pix[i, j][2] + rand
            if (a < 0):
                a = 0
            if (b < 0):
                b = 0
            if (c < 0):
                c = 0
            if (a > 255):
                a = 255
            if (b > 255):
                b = 255
            if (c > 255):
                c = 255
            draw.point((i, j), (a, b, c))
    image.save(save_path)
    t = np.array(image)
    del draw
    return t

def load_and_make_noisy_pictograms():
    picts = []
    picts_noisy = []
    for i in range(1,21):
        path = "picto/" + str(i) + ".png"
        pict = (tf.keras.preprocessing.image.load_img(path,color_mode="grayscale"))
        picts.append(tf.keras.preprocessing.image.img_to_array(pict))
        factor = 100
        picts_noisy.append(getNoisyBinaryImage(factor, path, "picto_test/noisy_" + str(factor) + "_" + str(i) + ".png"))

    return picts, picts_noisy


if __name__ == '__main__':
    picts, picts_noisy = load_and_make_noisy_pictograms()
    plt.imshow(picts[0],cmap=plt.cm.gray)
    plt.show()




