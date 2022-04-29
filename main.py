import pytesseract
from cv2 import cv2
from pytesseract import Output

IMAGE_FILE = "tests/fixtures/test1/1.1.scan-1.png"
annotation_file = "tests/fixtures/test2/1.2.scan-1.xml.json"


def character_recognition(image_file: str):
    """
    This method draws bounding boxes around each character and shows the modified file via cv2
    :param image_file:
    :return:
    """
    img = cv2.imread(image_file)

    height: int = img.shape[0]
    width: int = img.shape[1]

    imagedata = pytesseract.image_to_boxes(img, output_type=Output.DICT)
    word_count: int = len(imagedata['char'])

    for i in range(0, word_count):
        (text, x1, y2, x2, y1) = (
            imagedata['char'][int(i)], imagedata['left'][int(i)], imagedata['top'][int(i)], imagedata['right'][int(i)],
            imagedata['bottom'][int(i)])

        if len(text) != 1:
            raise RuntimeError("Tesseract recognized text with more than one character")

        cv2.rectangle(img, (x1, height - y1), (x2, height - y2), (0, 255, 0), 1)

    visualize(img)


def data_recognition(image_file: str):
    img = cv2.imread(image_file)
    imagedata = pytesseract.image_to_data(img, output_type=Output.DICT)
    boxes_count: int = len(imagedata['level'])

    for i in range(0, boxes_count):
        (x, y, width, height) = (
            imagedata['left'][i], imagedata['top'][i], imagedata['width'][i], imagedata['height'][i])
        text: str = imagedata['text'][i]

        # filter all empty strings and strings consisting of not alphanumeric characters
        if len(text.strip()) > 0 and not (len(text.strip()) == 1 and not text.strip().isalnum()):
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

    visualize(img)


def visualize(image):
    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img', image)
    cv2.resizeWindow('img', 900, 900)
    # waits for any keypress
    cv2.waitKey(0)


if __name__ == "__main__":
    # todo 2.2 text region recovery

    data_recognition(IMAGE_FILE)

    # load shared-file-format of corresponding file to be processed

    # load image file

    # recognize characters + their corresponding bounding boxes via tessaract or calamari

    # todo 2.3 cell recovery
