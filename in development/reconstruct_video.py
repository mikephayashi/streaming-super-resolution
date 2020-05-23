"""
Ref: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
FIXME: Frames out of order
"""
import torch

from Autoencoder import Autoencoder
# import cv2
# import os

# image_folder = './res/frames/test'
# video_name = 'video.avi'

# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width, height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()