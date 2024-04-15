# Author: Christian Galley
# date: March. 18, 2024

from PIL import Image
import numpy as np


def rgb_to_grayscale(image_path):
    image = np.array(Image.open(image_path))
    height, width, depth = image.shape
    result = np.zeros(shape=(height, width, depth), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i][j] = int(np.mean(image[i][j]))
    return result


# simple non-adaptative thresholding.
def algo1(image, threshold, final_name):
    height, width, depth = image.shape
    result = np.zeros(shape=(height, width, depth), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i][j][0] < threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255
    pil_img = Image.fromarray(result)
    pil_img.save(final_name)


def otsu(image, final_name):
    variance_list = []

    counts, bins = np.histogram(image, bins=255)
    counts = np.append(counts, [0])

    for i in range(len(bins)):
        p0 = sum(counts[:i]) / (sum(counts))
        p1 = sum(counts[i:]) / (sum(counts))
        u0 = 0
        u1 = 0

        for j in range(i):
            # print(str(i) + ": "+ str(counts[j]) + " * " + str(bins[j]))
            u0 += counts[j] * bins[j]

        for k in range(len(bins) - i):
            # print(str(i) + "(2): " + str(counts[i+k]) + " * " + str(bins[i+k]))
            u1 += counts[i + k] * bins[i + k]

        # avoid division by 0
        if u0 == 0:
            u0 = 0
        else:
            u0 = u0 / sum(counts[:i])
        if u1 == 0:
            u1 = 0
        else:
            u1 = u1 / sum(counts[i:])

        variance = p0 * p1 * (u0 - u1) ** 2
        variance_list.append(variance)

    threshold = variance_list.index(max(variance_list))

    algo1(image, threshold, final_name)


# binarisation based on local adaptative thresholding
def bernsen(image, r, l, bg, final_name):
    height, width, depth = image.shape
    result = np.zeros(shape=(height, width, depth), dtype=np.uint8)
    if bg == 'bright':
        k = 0
    else:
        k = 255
    for i in range(height):
        for j in range(width):
            elem = []
            for a in range(r):
                for b in range(r):
                    if i - 1 + a < 0 or j - 1 + b < 0:
                        pass
                    if i - 1 + a >= height or j - 1 + b >= width:
                        pass
                    else:
                        elem.append(image[i - 1 + a][j - 1 + b][0])

            elem_max = max(elem)
            elem_min = min(elem)
            threshold = (int(elem_max) + int(elem_min)) / 2
            contrast = int(elem_max) - int(elem_min)

            if contrast < l:
                wBTH = k
            else:
                wBTH = threshold
            if image[i][j][0] < wBTH:
                result[i][j] = 0
            else:
                result[i][j] = 255
    pil_img = Image.fromarray(result)
    pil_img.save(final_name)


grayscale_image = rgb_to_grayscale('aef-CSN-III-3-1_088-600x900.jpg')

algo1(grayscale_image, 127, 'binarization_simple.jpg')
otsu(image=grayscale_image, final_name='binarization_otsu.jpg')
bernsen(image=grayscale_image, r=7, l=30, bg="bright", final_name='binarization_bernsen.jpg')
