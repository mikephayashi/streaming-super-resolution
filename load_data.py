"""
Load Data ref: https://towardsdatascience.com/how-to-build-custom-dataloader-for-your-own-dataset-ae7fd9a40be6
"""

import sys
import getopt
import imageio
import torch
import numpy as np
from torch.utils import data


def pics_tensor(file_name, num_frames, width, height):
    """
    data_set = data.TensorDataset(pics_tensor("test", 11, 360, 480))
    data_loader = data.DataLoader(data_set) 
    """

    images = np.zeros((num_frames, width, height, 3))

    for i in range(0, num_frames):
        path = "./res/resized/{file_name}/frame{num}.jpg".format(
            file_name=file_name, num=i)
        images[i] = imageio.imread(path)

    return torch.tensor(images)


def print_args():
    print("python3 load_data.py -f/--file= <file name> -n/--number= <num frames> -w/--width=<width> -h/--height=<height>")


if __name__ == "__main__":

    name = "test"
    num_frames = 10

    argv = sys.argv[1:]
    file_name = None
    number = None
    width = None
    height = None

    try:
        opts, args = getopt.getopt(
            argv, "f:n:w:h:", ["file=", "number=", "width=", "height="])
    except getopt.GetoptError:
        print_args()

    for opt, arg in opts:
        if opt in ("-f", "--file"):
            file_name = arg
            if file_name is None:
                print_args()
                print("file is none")
                sys.exit()
        elif opt in ("-n", "--number"):
            number = int(arg)
            if number is None:
                print_args()
                print("number is none")
                sys.exit()
        elif opt in ("-w", "--width"):
            width = int(arg)
            if width is None:
                print_args()
                print("file is none")
                sys.exit()
        elif opt in ("-h", "--height"):
            height = int(arg)
            if height is None:
                print_args()
                print("height is none")
                sys.exit()


    images = pics_tensor(file_name, number, width, height)
    data_set = data.TensorDataset(images)
    data_loader = data.DataLoader(data_set)

    print(images.shape)
    print(type(data_set))
    print(type(data_loader))

