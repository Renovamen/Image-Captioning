from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from PIL import Image
import numpy as np

def visualize_att_beta(
    image_path: str,
    seq: list,
    alphas: list,
    rev_word_map: Dict[int, str],
    betas: list,
    smooth: bool = True
) -> None:
    """
    Visualize caption with weights and betas at every word.

    Parameters
    ----------
    image_path : str
        Path to image that has been captioned

    seq : list
        Generated caption on the above mentioned image using beam search

    alphas : list
        Attention weights at each time step

    betas : list
        Sentinel gate at each time step (only in 'adaptive_att' mode)

    rev_word_map : Dict[int, str]
        Reverse word mapping, i.e. ix2word

    smooth : bool, optional, default=True
        Smooth weights or not?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # subplot settings
    num_col = len(words) - 1
    num_row = 1
    subplot_size = 4

    # graph settings
    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * num_row)

    img_size = 4
    fig_height = img_size
    fig_width = num_col + img_size

    grid = plt.GridSpec(fig_height, fig_width)

    # big image
    plt.subplot(grid[0 : img_size, 0 : img_size])
    plt.imshow(image)
    plt.axis('off')

    # betas' curve
    if betas is not None:
        plt.subplot(grid[0 : fig_height - 1, img_size : fig_width])

        x = range(1, len(words), 1)
        y = [ (1 - betas[t].item()) for t in range(1, len(words)) ]

        for a, b in zip(x, y):
            plt.text(a + 0.05, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=12)

        plt.axis('off')
        plt.plot(x, y)

    for t in range(1, len(words)):
        if t > 50:
            break

        plt.subplot(grid[fig_height - 1, img_size + t - 1])

        # images
        plt.imshow(image)

        # words of sentence
        plt.text(0, 500, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=10)

        # alphas
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        plt.imshow(alpha, alpha=0.6)
        plt.set_cmap('jet')

        plt.axis('off')

    plt.show()

def visualize_att(
    image_path: str,
    seq: list,
    alphas: list,
    rev_word_map: Dict[int, str],
    smooth: bool = True
) -> None:
    """
    Visualize caption with weights at every word.

    Adapted from: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    Parameters
    ----------
    image_path : str
        Path to image that has been captioned

    seq : list
        Generated caption on the above mentioned image using beam search

    alphas : list
        Attention weights at each time step

    rev_word_map : Dict[int, str]
        Reverse word mapping, i.e. ix2word

    smooth : bool, optional, default=True
        Smooth weights or not?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    # subplot settings
    num_col = 5
    num_row = np.ceil(len(words) / float(num_col))
    subplot_size = 4

    # graph settings
    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * num_row)

    for t in range(len(words)):
        if t > 50:
            break

        plt.subplot(num_row, num_col, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)

        plt.imshow(image)

        current_alpha = alphas[t, :]

        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])

        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)

        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.show()
