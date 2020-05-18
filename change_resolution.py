"""
change_resolution.py
Michael Hayashi

Given a vidoe, outputs frames at changed resolution
Outputs to `res/frames/<name>` (extracted frames of video) and to `res/resized/<name>` (changed resolution of extractedframes)
"""

import sys
import os
import cv2
import sys
import getopt
from PIL import Image

class Change_Resolution:

    def __init__(self, name, video):
        self.video = video
        self.extracted_path = "./res/frames/" + name + "/"
        self.resized_path = "./res/resized/" + name + "/"
        self.info_path = "./res/info/" + name + ".txt"
        self.num_frames = 0
        if not os.path.exists(self.extracted_path):
            os.makedirs(self.extracted_path)
        if not os.path.exists(self.resized_path):
            os.makedirs(self.resized_path)

        # Output picture dimensions
        with open(self.info_path, 'a+') as info:
            info.write("---------------\n")
            info.write("Change Resolution {name}\n".format(name=name))
            info.close()
        

    def extract_frames(self):
        """
        Extracts frames from videos
        """
        vidcap = cv2.VideoCapture(self.video)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(self.extracted_path + "frame%d.jpg" %
                        self.num_frames, image)     # save frame as JPEG file 
            success, image = vidcap.read()
            print('Read a new frame: ', success, ": " count)
            self.num_frames += 1
            count += 1

        with open(self.info_path, 'a+') as info:
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
            info.write("Original dimensions {w}x{h}\n".format(w=width, h=height))
            info.close()

    def change_res(self, width, height):
        """
        Changes frame resolution
        """
        for i in range(0, self.num_frames):
            image = Image.open(self.extracted_path + "frame%d.jpg" % i)
            resized_image = image.resize((width, height))
            resized_image.save(self.resized_path + "frame%d.jpg" % i)
            print('Changing res: image ', i)
        
        with open(self.info_path, 'a+') as info:
            info.write("New dimensions {w}x{h}\n".format(w=width, h=height))
            info.write("---------------\n")
            info.close()


def print_args():
    print("Usage: python3 change_resolution.py -n <name> -v <video> [optional]-w <width> [optional]-h <height>")


if __name__ == "__main__":
    argv = sys.argv[1:]

    name = None
    video = None
    width = 100
    height = 100

    try:
        opts, args = getopt.getopt(argv, "n:v:w:h:", ["name=", "video=", "width=", "height="])
    except getopt.GetoptError:
        print_args()

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            name = arg
        elif opt in ("-v", "--video"):
            video = arg
        elif opt in ("-w", "--width"):
            width = arg
        elif opt in ("-h", "--height"):
            height = arg
    
    if name is None or video is None:
        print_args()
        sys.exit()
        
    original_vid = Change_Resolution(name, video)
    original_vid.extract_frames()
    original_vid.change_res(width, height)


