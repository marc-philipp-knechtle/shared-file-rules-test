import random
import sys

import pytesseract
from cv2 import cv2
from pytesseract import Output
from pandas import DataFrame

from loguru import logger

import copy

logger.remove()
logger.add(sys.stderr, level="DEBUG")

IMAGE_FILE = "tests/fixtures/images/1.6.scan-1.png"
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

        logger.info(text)

        if len(text) != 1:
            raise RuntimeError("Tesseract recognized text with more than one character")

        cv2.rectangle(img, (x1, height - y1), (x2, height - y2), (0, 255, 0), 1)

    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img', img)
    cv2.resizeWindow('img', 900, 900)
    # waits for any keypress
    cv2.waitKey(0)


def data_recognition(image_file: str):
    img = cv2.imread(image_file)

    # dict image data contains of the following keys
    # level = level of the box: possible values: (1)page, (2)block, (3)paragraph, (4)line, (5)word
    # page_num = page number
    # block_num = ?
    # par_num = ?
    # word_num = ?
    # left = x coordinate
    # top = y coordinate
    # width
    # height
    # conf = ?
    # text = content
    # line_num = ?
    imagedata: dict = pytesseract.image_to_data(img, output_type=Output.DICT)
    # todo write this with pandas dataframe instead of simple dictionary
    image_as_dataframe: DataFrame = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)

    original_imagedata_dict = copy.deepcopy(imagedata)
    imagedata = filter_empty_imagedata(imagedata)
    imagedata = merge_to_text_blocks(imagedata, count_of_called=1)

    visualize(img, imagedata)
    visualize(img, original_imagedata_dict)


def filter_empty_imagedata(imagedata: dict) -> dict:
    indexes_to_remove = []
    for i in range(len(imagedata['level'])):
        text: str = imagedata['text'][i]
        if len(text.strip()) > 0 and not (len(text.strip()) == 1 and not text.strip().isalnum()):
            continue
        else:
            indexes_to_remove.append(i)

    # remove all duplicate values by converting to set
    indexes_to_remove: set = set(indexes_to_remove)
    # remove higher indexes first to not be conflicted with loop and index removal
    for index in sorted(indexes_to_remove, reverse=True):
        imagedata = remove_index(imagedata, index)

    return imagedata


def merge_to_text_blocks(imagedata: dict, count_of_called: int) -> dict:
    """
    This method correlates to Section 2.2. in shigarov2016configurable
    :return: the merged bounding boxes
    """
    logger.info("Called merge_to_text_blocks for the " + str(count_of_called) + "th time.")

    boxes_count: int = len(imagedata['level'])

    indexes_to_remove = []

    p_1_count: int = 0

    for i in range(0, boxes_count):
        # indexes already to be removed will be filtered out in the next stage.
        if i in indexes_to_remove:
            continue
        (x1, y1, width1, height1) = (
            imagedata['left'][i], imagedata['top'][i], imagedata['width'][i], imagedata['height'][i])
        text1: str = imagedata['text'][i]

        for j in range(0, boxes_count):
            # don't compare equal indexes + don't compare indexes which are removed
            # these who would theoretically match to those indexes will be filtered out in the next stage
            if j == i or j in indexes_to_remove:
                continue
            (x2, y2, width2, height2) = (
                imagedata['left'][j], imagedata['top'][j], imagedata['width'][j], imagedata['height'][j])
            text2: str = imagedata['text'][j]

            if p_1_word_spacing(y1, y2) \
                    and p_2_vertical_projections(imagedata, x1, width1, x2, width2):
                logger.debug("Merging bboxes with content text1: [" + text1 + "] and text2: [" + text2 + "]")
                p_1_count += 1

                merged_box: list = _merge_boxes([x1, y1, width1, height1], [x2, y2, width2, height2])
                indexes_to_remove.append(i)
                indexes_to_remove.append(j)
                imagedata['left'].append(merged_box[0])
                imagedata['top'].append(merged_box[1])
                imagedata['width'].append(merged_box[2])
                imagedata['height'].append(merged_box[3])
                imagedata['text'].append(text1 + text2)
                # todo change this with a proper mechanism
                imagedata['level'].append(1)
                imagedata['page_num'].append(1)
                imagedata['block_num'].append(1)
                imagedata['par_num'].append(1)
                imagedata['word_num'].append(1)
                imagedata['conf'].append(1)
                imagedata['line_num'].append(1)

                # calling break here to avoid the creation of bounding box combination which already exist
                # in another form
                # These will filter out if that method has been run multiple times.
                break

    # remove all duplicates
    indexes_to_remove: set = set(indexes_to_remove)
    for index in sorted(indexes_to_remove, reverse=True):
        imagedata = remove_index(imagedata, index)

    logger.info("Performed " + str(p_1_count) + " P_1 merge operations.")

    if len(indexes_to_remove) != 0:
        imagedata = merge_to_text_blocks(imagedata, count_of_called=count_of_called + 1)

    return imagedata


def p_1_word_spacing(y1, y2) -> bool:
    """
    See corresponding rule in shigarov2016configurable
    on page 2

    # todo problem of this approach
    overlapping bounding boxes. The bounding boxes around words are not exact. There are often overlapping bboxes
    e.g. the bbox from two different cells are crossing -> it's not clear where the bounding box is perfekt and where not

    :return: true if those two boxes should be merged according to rule p1
    """
    # todo make this absolute value to a relative value compared to image height
    # question: is a relative to total size index useful?
    # no - this value is also influenced by the size of the table
    # the cells may mave a smaller difference to their neighbour (if taken relative to the image size) if the table is big
    # maybe make this value dependent on the general font size
    if abs(y1 - y2) < 50:
        return True
    return False


def p_2_vertical_projections(imagedata: dict, x_1, width_1, x_2, width_2) -> bool:
    """
    see p2 in shigarov2016configurable

    p2 from paper:
    there is a configurable intersection of their vertical projections

    from corresponding documentation in tabbypdf:
    Vergewissern sie sich dass der erste Teil links vom zweiten liegt + sich deren projektionen
    (in die vertikale richtung) überschneiden.

    todo
    why is it necessary that the first part is left of the right?  todo currently not implemented

    why is it necessary that there is a intersection of their projections?
    if there arent any common projection elements - these are elements which are horizontically bound together.
    These could be theoretically at the other end of the table.

    :return:
    """
    vertical_projections_1: list = _get_vertical_projections(imagedata, x_1, width_1)
    vertical_projections_2: list = _get_vertical_projections(imagedata, x_2, width_2)

    if not set(vertical_projections_1).isdisjoint(vertical_projections_2):
        return True
    return False


def _get_vertical_projections(imagedata: dict, x: int, width: int) -> list:
    vertical_projections: list = []
    range_to_match = range(x, x + width)
    for i in range(len(imagedata['level'])):
        x_1: int = imagedata['left'][i]
        width_1: int = imagedata['width'][i]

        if _range_subset(range(x_1, x_1 + width_1), range_to_match):
            vertical_projections.append(i)

    return vertical_projections


def _range_subset(range1: range, range2: range) -> bool:
    """
    Checks whether range1 is a partial subset of range2.

    :param range1:
    :param range2:
    :return:
    """
    if not range1:
        return True
    if not range2:
        return False

    for element in range1:
        if element in range2:
            return True

    return False


def p_3_line_spacing():
    pass


def p_4_horizontal_projections():
    pass


def _merge_boxes(box1, box2) -> list:
    """
    box content (at index)
    1. x coordinate
    2. y coordinate
    3. width
    4. height
    :param box1:
    :param box2:
    :return:
    """
    box_1_point_1 = (box1[0], box1[1])
    box_1_point_2 = (box1[0] + box1[2], box1[1] + box1[3])
    box_2_point_1 = (box2[0], box2[1])
    box_2_point_2 = (box2[0] + box2[2], box2[1] + box2[3])

    # box 1 lower x + box 2 lower x
    lower_new_x: int = min(box_1_point_1[0], box_2_point_1[0])
    # box 1 lower y + box 2 lower y
    lower_new_y: int = min(box_1_point_1[1], box_2_point_1[1])
    # box 1 upper x + box 2 upper x
    upper_new_x: int = max(box_1_point_2[0], box_2_point_2[0])
    # box 1 upper y + box 2 upper y
    upper_new_y: int = max(box_1_point_2[1], box_2_point_2[1])

    width = abs(upper_new_x - lower_new_x)
    height = abs(upper_new_y - lower_new_y)

    return [lower_new_x,
            lower_new_y,
            width,
            height]


def remove_index(dct: dict, index: int) -> dict:
    """
    specially written for the dictionary returned by pytesseract.
    this removes the entry for every key at index xy
    :return: modified dict
    """
    logger.debug("Called to delete element at index: " + str(index))
    for key in dct:
        del dct[key][index]

    return dct


def draw_bounding_boxes_from_dict(imagedata, img):
    boxes_count: int = len(imagedata['level'])
    for i in range(0, boxes_count):
        (x, y, width, height) = (
            imagedata['left'][i], imagedata['top'][i], imagedata['width'][i], imagedata['height'][i])
        text: str = imagedata['text'][i]

        # filter all empty strings and strings consisting of not alphanumeric characters
        if len(text.strip()) > 0 and not (len(text.strip()) == 1 and not text.strip().isalnum()):
            cv2.rectangle(img, (x, y), (x + width, y + height), random.choices(range(256), k=3), 2)

    return img


def visualize(image, imagedata):
    image = draw_bounding_boxes_from_dict(imagedata, image)

    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img', image)
    cv2.resizeWindow('img', 900, 900)
    # waits for any keypress
    cv2.waitKey(0)

    cv2.destroyWindow('img')


if __name__ == "__main__":
    # todo 2.2 text region recovery

    data_recognition(IMAGE_FILE)

    # load shared-file-format of corresponding file to be processed

    # load image file

    # recognize characters + their corresponding bounding boxes via tessaract or calamari

    # todo 2.3 cell recovery
