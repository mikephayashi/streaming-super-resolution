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
import shutil
from PIL import Image
from os import listdir
from os.path import isfile, join
import threading
from threading import Semaphore

class Change_Resolution:

    def __init__(self, name, video, skip):
        self.video = video
        self.extracted_path = "./res/frames/" + name + "/"
        self.resized_path = "./res/resized/" + name + "/"
        self.info_path = "./res/info/" + name + ".txt"
        self.num_frames = 0
        self.skip = skip
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
        to_skip = self.skip
        while success:
            if to_skip == 0:
                cv2.imwrite(self.extracted_path + "frame%d.jpg" %
                            self.num_frames, image)     # save frame as JPEG file 
                self.num_frames += 1
            success, image = vidcap.read()
            to_skip -= 1
            if to_skip < 0:
                to_skip = self.skip

        with open(self.info_path, 'a+') as info:
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
            info.write("Num of frames: {count}\n".format(count=self.num_frames))
            info.write("Original dimensions {w}x{h}\n".format(w=width, h=height))
            info.close()

    def change_res(self, width, height):
        """
        Changes frame resolution
        """
        # FIXME: NUM OF SKIP FRAMES
        for i in range(0, self.num_frames):
            image = Image.open(self.extracted_path + "frame%d.jpg" % i)
            resized_image = image.resize((width, height))
            resized_image.save(self.resized_path + "frame%d.jpg" % i)
            # print('Changing res: image ', i)
        
        with open(self.info_path, 'a+') as info:
            info.write("New dimensions {w}x{h}\n".format(w=width, h=height))
            info.write("---------------\n")
            info.close()

    def remove_vid(self):
        # shutil.rmtree(self.extracted_path)
        os.remove(self.video)


def print_args():
    print("Usage: python3 change_resolution.py -n <name> -v <video> [optional]-w <width> [optional]-h <height> -s/--skip= <num to skip>")


if __name__ == "__main__":
    argv = sys.argv[1:]

    name = None
    video = None
    width = 256
    height = 256
    skip = 6

    try:
        opts, args = getopt.getopt(argv, "n:v:w:h:s:", ["name=", "video=", "width=", "height=", "skip="])
    except getopt.GetoptError:
        print_args()

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            name = arg
        elif opt in ("-v", "--video"):
            video = arg
        elif opt in ("-w", "--width"):
            width = int(arg)
        elif opt in ("-h", "--height"):
            height = int(arg)
        elif skip in ("-s", "--skip"):
            skip = int(arg)

    # if name is None or video is None:
    #     print_args()
    #     sys.exit()

    # Make necessary dirs
    if not os.path.exists("./res"):
        os.makedirs("./res")
    if not os.path.exists("./res/info"):
        os.makedirs("./res/info")

    # Thread control
    sem = Semaphore(10)  

    onlyfiles = [f for f in listdir("./res/youtube_vids") if isfile(join("./res/youtube_vids", f)) and f != ".DS_Store"]

    def get_imgs(file_name):
        sem.acquire()
        print("Changing resolution {file_name}".format(file_name=file_name))
        original_vid = Change_Resolution(file_name, "./res/youtube_vids/{file_name}".format(file_name=file_name), skip)
        original_vid.extract_frames()
        # original_vid.change_res(width, height)
        original_vid.remove_vid()
        
        sem.release()
    
    threads = []
    for file_name in onlyfiles:
        thread = threading.Thread(target=get_imgs, args=(file_name,))
        threads.append(thread)
        thread.start()

    for index, thread in enumerate(threads):
        thread.join()
