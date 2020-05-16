import sys
from PIL import Image


class Change_Resolution:

    def __init__(self, image):
        self.image = Image.open("./res/" + image + ".jpg")
        #  self.image = Image.open(image)

    def change_res(self, width, height):
        resized_image = self.image.resize((width, height))
        resized_image.save("./resized/test-" + str(width) + "x" + str(height) + ".png")


if __name__ == "__main__":
    input_image = sys.argv[1]
    original_image = Change_Resolution(input_image)
    original_image.change_res(100, 100)