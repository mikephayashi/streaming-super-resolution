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

    def __init__(self, name, data_type, video, skip):
        self.video = video
        self.data_type = data_type
        self.extracted_train_path = "./res/frames/train/" + name + "/"
        self.extracted_test_path = "./res/frames/test/" + name + "/"
        # self.resized_path = "./res/resized/" + name + "/"
        self.info_path = "./res/info/" + name + ".txt"
        self.num_frames = 0
        self.skip = skip
        # if not os.path.exists(self.resized_path):
        #     os.makedirs(self.resized_path)
        if data_type == "train" and not os.path.exists(self.extracted_train_path):
            os.makedirs(self.extracted_train_path)
        if data_type == "test" and not os.path.exists(self.extracted_test_path):
            os.makedirs(self.extracted_test_path)

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
                print("dt: ", self.data_type)
                if self.data_type == "train":
                    cv2.imwrite(self.extracted_train_path + "frame%d.jpg" %
                                self.num_frames, image)     # save frame as JPEG file
                elif self.data_type == "test":
                    cv2.imwrite(self.extracted_test_path + "frame%d.jpg" %
                                self.num_frames, image)
                self.num_frames += 1
            success, image = vidcap.read()
            to_skip -= 1
            if to_skip < 0:
                to_skip = self.skip

        with open(self.info_path, 'a+') as info:
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            info.write("Num of frames: {count}\n".format(
                count=self.num_frames))
            info.write("Original dimensions {w}x{h}\n".format(
                w=width, h=height))
            info.close()

    def change_res(self, width, height):
        """
        Changes frame resolution
        """
        # FIXME: NUM OF SKIP FRAMES
        for i in range(0, self.num_frames):
            image = Image.open(self.extracted_train_path + "frame%d.jpg" % i)
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
    print(
        "Usage: python3 change_resolution.py [optional]-w <width> [optional]-h <height> -s/--skip= <num to skip default=6>")


if __name__ == "__main__":
    argv = sys.argv[1:]

    width = 256
    height = 256
    skip = 6

    try:
        opts, args = getopt.getopt(
            argv, ":w:h:s:", ["width=", "height=", "skip="])
    except getopt.GetoptError:
        print_args()

    for opt, arg in opts:
        if opt in ("-w", "--width"):
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
    sem = Semaphore(3)

    train_vids = [f for f in listdir("./res/youtube_vids/train") if isfile(
        join("./res/youtube_vids/train", f)) and f != ".DS_Store"]
    test_vids = [f for f in listdir("./res/youtube_vids/test") if isfile(
        join("./res/youtube_vids/test", f)) and f != ".DS_Store"]

    def get_imgs(file_name, data_type):
        sem.acquire()
        print("Extracting frames {file_name}".format(file_name=file_name))
        original_vid = Change_Resolution(
            file_name, data_type, "./res/youtube_vids/{data_type}/{file_name}".format(data_type=data_type, file_name=file_name), skip)
        original_vid.extract_frames()
        # original_vid.change_res(width, height)
        # original_vid.remove_vid()

        sem.release()

    threads = []

    def get_vids(data_type, vid_type):
        for file_name in vid_type:
            thread = threading.Thread(
                target=get_imgs, args=(file_name, data_type,))
            threads.append(thread)
            thread.start()

    get_vids("train", train_vids)
    get_vids("test", test_vids)

    for index, thread in enumerate(threads):
        thread.join()
