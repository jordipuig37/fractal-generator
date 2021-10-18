import numpy as np
from PIL import Image
from numpy import array
import colorsys
from time import time
from tqdm import tqdm
import argparse
import copy

from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib import cm

from aux_func import *

MAX_ITER = 2048


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalizes

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def julia_idx_from_idx(i, j, args, func):
    cx, cy = args.cx, args.cy
    SIZE = args.size
    x = (i - SIZE/2) / (SIZE/args.zoom)
    y = (j - SIZE/2) / (SIZE/args.zoom)
    z = complex(x, y)
    c = complex(cx, cy)
    for k in range(1, MAX_ITER):
        if abs(z) > 4:
            return k
        z = func(z, c)
    return 0


def get_func(args):
    if args.function == "power2":
        return power2
    elif args.function == "power3":
        return power3
    elif args.function == "power4":
        return power4
    elif args.function == "power5":
        return power5
    elif args.function == "exp_power3":
        return exp_power3
    elif args.function == "frac_z_lnz":
        return frac_z_lnz
    else:
        return power2


def parallel_julia(args):
    SIZE = args.size
    # creating the new image in RGB mode
    colormap = cm.get_cmap(args.cm, MAX_ITER)

    func = get_func(args)

    with Pool(args.threads, maxtasksperchild=1000) as p:
        tots_els_px = list()
        for i in tqdm(range(SIZE)):
            points = [(i, j, args, func) for j in range(SIZE)]
            row = p.starmap(julia_idx_from_idx, points)
            tots_els_px.append(copy.deepcopy(row))

    pixels = np.array(tots_els_px)
    # give cool color to the indexes
    grayscale = pixels

    if args.equalize:
        equalized, _ = image_histogram_equalization(grayscale, MAX_ITER)
        normalized = equalized / equalized.max()
    else:
        normalized = grayscale

    cx, cy = args.cx, args.cy
    args_str = f"{cx:.02}_{cy:.02}_{args.function}_{SIZE}_{args.cm}"
    
    img = Image.fromarray(np.uint8(colormap(normalized)*255))
    img.save(f"{args.dest_path}/{args_str}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cx", type=float, default=0.0)
    parser.add_argument("-cy", type=float, default=0.0)

    parser.add_argument("-function", type=str, default="power2")
    parser.add_argument('--equalize', dest='equalize', action='store_true')
    parser.add_argument('--no-equalize', dest='equalize', action='store_false')
    parser.set_defaults(equalize=True)
    
    parser.add_argument("-size", type=int, default=512)
    parser.add_argument("-zoom", type=float, default=2)
    parser.add_argument("-cm", type=str, default="viridis")

    parser.add_argument("-threads", type=int, default=1)
    parser.add_argument("--dest-path", type=str, default=".")

    args = parser.parse_args()

    parallel_julia(args)