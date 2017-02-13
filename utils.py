import matplotlib
import inspect
import cv2
import numpy as np

from settings import DEBUG, DISPLAY

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt


def debug(*args):
    frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
    if DEBUG:
        print('[%s:%d]' % (function_name, line_number), *args)


def display(image, msg='Image', cmap=None):
    if not DISPLAY: return

    if image.ndim == 2:
        cmap = 'gray'

    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(msg, fontsize=30)
    plt.show(block=True)


def imcompare(image1, image2, msg1='Image1', msg2='Image2', cmap1=None, cmap2=None):
    if DISPLAY == False: return

    if image1.ndim == 2:
        cmap1 = 'gray'
    if image2.ndim == 2:
        cmap2 = 'gray'

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    ax1.imshow(image1, cmap=cmap1)
    ax1.set_title(msg1, fontsize=30)
    ax2.imshow(image2, cmap=cmap2)
    ax2.set_title(msg2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # title = f.suptitle(msg1)
    f.tight_layout()
    # title.set_y(0.75)
    plt.show(block=True)


def warper(img, src, dst, flip=True):
    # Compute and apply perpective transform
    if flip:
        # Resultant image (h,w) = (w,h) of input `img`
        # import ipdb; ipdb.set_trace()
        img_size = (img.shape[0], img.shape[1])
        w, h = img_size
        w_padding, h_padding = w*0.0, h*0.0

        dst = np.array([[0+w_padding, 0+h_padding],
                        [w-w_padding, 0+h_padding],
                        [w-w_padding, h-h_padding],
                        [0+w_padding, h-h_padding]], np.float32)
    else:
        # Resultant image keeps the (h,w) of input `img`
        img_size = (img.shape[1], img.shape[0])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def dstack(binary1, binary2):
    """
        Stack each channel to view their individual contributions in green and blue respectively
        This returns a stack of the two binary images, whose components you can see as different colors
    """
    color_binary = np.dstack((np.zeros_like(binary1), binary1, binary2))
    return color_binary


def hist(img):
    color = ('r', 'g', 'b')
    plt.figure()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        plt.plot(histr, color=col)
        plt.xlim([-1, 3])
    plt.show()
