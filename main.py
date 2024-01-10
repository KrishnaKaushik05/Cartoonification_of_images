import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_file(filename):
    img = cv2.imread(filename)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        plt.imshow(img, cmap='gray')  # Display grayscale image
        plt.show()
        return img
    else:
        print("Error: Image not loaded successfully")
        return None

def edge_mark(img, line_size, blur_value):
    gray_blur = cv2.medianBlur(img, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 1))  # Reshape to a column vector
    data = np.repeat(data, 3, axis=1)  # Replicate grayscale values across 3 channels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape + (3,))  # Reshape to 3-channel color shape
  # Reshape back to original image shape
    return result





filename = 'Tger_image.jpg'
img = read_file(filename)

line_size, blur_value = 7, 7
edges = edge_mark(img, line_size, blur_value)
plt.imshow(edges, cmap='gray')
plt.show()

img = color_quantization(img, k=3)
plt.imshow(img)
plt.show()
blurred = cv2.bilateralFilter(img,d =7,sigmaColor=200,sigmaSpace=200)
plt.imshow(blurred)
plt.show()

def cartoon():
    c = cv2.bitwise_and(blurred,blurred,mask=edges)
    plt.imshow(c)
    plt.show()
cartoon()