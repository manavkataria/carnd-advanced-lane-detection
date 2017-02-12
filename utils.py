import matplotlib
import inspect

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DISPLAY = False
DEBUG = True


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
    f.tight_layout()
    ax1.imshow(image1, cmap=cmap1)
    ax1.set_title(msg1, fontsize=30)
    ax2.imshow(image2, cmap=cmap2)
    ax2.set_title(msg2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show(block=True)
