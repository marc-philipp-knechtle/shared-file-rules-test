import os
import random
import sys

import pytesseract
from cv2 import cv2
from pytesseract import Output

from loguru import logger

import copy

logger.remove()
logger.add(sys.stderr, level="DEBUG")

IMAGE_FILE = "tests/fixtures/images/1.6.scan-1.png"
annotation_file = "tests/fixtures/test2/1.2.scan-1.xml.json"

P_1_HORIZONTAL_WORD_SPACING_DISTANCE: int = 0  # distance of words next to each other
P_3_VERTICAL_LINE_SPACING_DISTANCE: int = 0

P_1_DIVISION_FACTOR: float = 0.5
P_3_DIVISION_FACTOR: float = 0.5

# todo idea: make all detected cells white and run this algorithm again -> new detection of cells?


def character_recognition(image_file: str):
    """
    This method draws bounding boxes around each character and shows the modified file via cv2
    :param image_file:
    :return:
    """
    img = cv2.imread(image_file)

    height: int = img.shape[0]
    # width: int = img.shape[1]

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


def data_recognition(imagedata: dict, img):
    logger.warning("The Recognition has the (x=0, y=0) coordinates at the Top Left.")
    original_imagedata_dict = copy.deepcopy(imagedata)
    original_image = copy.deepcopy(img)

    imagedata = filter_empty_imagedata(imagedata)
    imagedata = merge_to_text_blocks(imagedata, count_of_called=1)

    visualize_original_and_processed(original_image, img, original_imagedata_dict, imagedata)


def filter_empty_imagedata(imagedata: dict) -> dict:
    indexes_to_remove = []
    for i in range(len(imagedata['level'])):
        text: str = imagedata['text'][i]
        if len(text.strip()) > 0 and not (len(text.strip()) == 1 and not text.strip().isalnum()):
            continue
        else:
            indexes_to_remove.append(i)
            logger.debug("Removing element at index: [" + str(i) + "] because of empty text representation.")

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
    p_3_count: int = 0

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

            if p_1_word_spacing(x1, x2, width1, width2) \
                    and p_2_vertical_projections(imagedata, x1, width1, x2, width2) \
                    and p_1_2_same_y_level(y1, y2, height1, height2):
                logger.debug(
                    "Merging bboxes with content text1: [" + text1 + "] and "
                                                                     "text2: [" + text2 + "] because of p_1 and p_2")
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

            if p_3_line_spacing(y1, y2, height1, height2) \
                    and p_4_horizontal_projections(imagedata, y1, height1, y2, height2) \
                    and p_3_4_same_x_level(x1, x2, width1, width2):
                logger.debug(
                    "Merging bboxes with content text1: [" + text1 + "] and "
                                                                     "text2: [" + text2 + "] because of p_3 and p_4")

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

                p_3_count += 1

                break

    # remove all duplicates
    indexes_to_remove: set = set(indexes_to_remove)
    for index in sorted(indexes_to_remove, reverse=True):
        imagedata = remove_index(imagedata, index)

    logger.info("Performed " + str(p_1_count) + " P_1/P_2 merge operations.")
    logger.info("Performed " + str(p_3_count) + " P_3/P_4 merge operations.")

    if len(indexes_to_remove) != 0:
        imagedata = merge_to_text_blocks(imagedata, count_of_called=count_of_called + 1)

    return imagedata


def p_3_line_spacing(y1, y2, height1, height2) -> bool:
    """
    See corresponding rule in shigarov2016configurable
    on page 2

    # todo problem of this approach
    overlapping bounding boxes. The bounding boxes around words are not exact. There are often overlapping bboxes
    e.g. the bbox from two different cells are crossing
    -> it's not clear where the bounding box is perfekt and where not

    :return: true if those two boxes should be merged according to rule p1
    """
    # todo make this absolute value to a relative value compared to image height
    # question: is a relative to total size index useful?
    # no - this value is also influenced by the size of the table
    # the cells may mave a smaller difference to their neighbour
    # (if taken relative to the image size) if the table is big
    # maybe make this value dependent on the general font size

    # cell 1 is above cell 2
    if abs((y2 - height2) - y1) < P_3_VERTICAL_LINE_SPACING_DISTANCE:
        return True
    # cell 2 is above cell1
    if abs((y1 - height1) - y2) < P_3_VERTICAL_LINE_SPACING_DISTANCE:
        return True
    return False


def p_2_vertical_projections(imagedata: dict, x_1, width_1, x_2, width_2) -> bool:
    # todo create the projections with the angle of the coordinate
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

    anschaulich: was bedeutet es für hoeizontale zusammenfügung wenn eine vertikale zelle als peojektion existiert?
    -> die beiden die zusammengefügt werden kommen save aus einer zelle

    :return:
    """
    vertical_projections_1: list = _get_vertical_projections(imagedata, x_1, width_1)
    vertical_projections_2: list = _get_vertical_projections(imagedata, x_2, width_2)

    if not set(vertical_projections_1).isdisjoint(vertical_projections_2):
        return True
    return False


def p_1_word_spacing(x1, x2, width1: int, width2: int) -> bool:
    # this is only the case if the bboxes are relatively small because the distance-difference to the right is not as
    # much when considering the with of the cell
    if abs(x1 - x2) < P_1_HORIZONTAL_WORD_SPACING_DISTANCE:
        return True
    # x1 left of x2 and already merged
    if abs((x1 + width1) - x2) < P_1_HORIZONTAL_WORD_SPACING_DISTANCE:
        return True
    # x2 is left of x1
    if abs((x2 + width2) - x1) < P_1_HORIZONTAL_WORD_SPACING_DISTANCE:
        return True
    return False


def p_1_2_same_y_level(y1, y2, height1, height2) -> bool:
    range_1: range = range(y1, y1 + height1)
    range_2: range = range(y2, y2 + height2)

    if len(set(range_1).intersection(range_2)) > (height1 + height2) / 4:
        return True
    return False


def p_3_4_same_x_level(x1, x2, width1, width2) -> bool:
    range_1: range = range(x1, x1 + width1)
    range_2: range = range(x2, x2 + width2)

    if len(set(range_1).intersection(range_2)) > (width1 + width2) / 4:
        return True
    return False


def p_4_horizontal_projections(imagedata: dict, y_1, height_1, y_2, height_2) -> bool:
    horizontal_projections_1: list = _get_horizontal_projections(imagedata, y_1, height_1)
    horizontal_projections_2: list = _get_horizontal_projections(imagedata, y_2, height_2)

    if not set(horizontal_projections_1).isdisjoint(horizontal_projections_2):
        return True
    return False


def _get_vertical_projections(imagedata: dict, x: int, width: int) -> list:
    vertical_projection_indexes: list = []
    range_to_match: range = range(x, x + width)
    for i in range(len(imagedata['level'])):
        x_1: int = imagedata['left'][i]
        width_1: int = imagedata['width'][i]

        if _range_subset(range(x_1, x_1 + width_1), range_to_match):
            vertical_projection_indexes.append(i)

    return vertical_projection_indexes


def _get_horizontal_projections(imagedata: dict, y: int, height: int) -> list:
    horizontal_projection_indexes: list = []
    range_to_match: range = range(y, y - height)
    for i in range(len(imagedata['level'])):
        y_1: int = imagedata['top'][i]
        height_1: int = imagedata['height'][i]

        if _range_subset(range(y_1, y_1 + height_1), range_to_match):
            horizontal_projection_indexes.append(i)

    return horizontal_projection_indexes


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

    # noinspection PyTypeChecker
    width = abs(upper_new_x - lower_new_x)
    # noinspection PyTypeChecker
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


def visualize_original_and_processed(original_image, processed_image, imagedata_original, imagedata_processed):
    image_original = draw_bounding_boxes_from_dict(imagedata_original, original_image)
    image_processed = draw_bounding_boxes_from_dict(imagedata_processed, processed_image)

    cv2.namedWindow('original', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('original', image_original)
    cv2.resizeWindow('original', 900, 900)

    cv2.namedWindow('processed', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('processed', image_processed)
    cv2.resizeWindow('processed', 900, 900)

    # waits for any keypress
    cv2.waitKey(0)
    cv2.destroyWindow('original')


def get_average_cell_width(imagedata: dict) -> int:
    imagedata_filtered = filter_empty_imagedata(imagedata)
    total_width: int = 0
    for i in range(len(imagedata_filtered['level'])):
        total_width += imagedata_filtered['width'][i]
    # noinspection PyTypeChecker
    return round(total_width / len(imagedata_filtered['level']))


def get_average_cell_height(imagedata: dict) -> int:
    imagedata_filtered = filter_empty_imagedata(imagedata)
    total_height: int = 0
    for i in range(len(imagedata_filtered['level'])):
        total_height += imagedata_filtered['height'][i]
    # noinspection PyTypeChecker
    return round(total_height / len(imagedata_filtered['level']))


def determine_horizontal_spacing_distance(imagedata: dict) -> int:
    average_cell_width = get_average_cell_width(imagedata)
    # noinspection PyTypeChecker
    return round(average_cell_width / P_3_DIVISION_FACTOR)


def determine_vertical_spacing_distance(imagedata: dict) -> int:
    average_cell_height = get_average_cell_height(imagedata)
    # noinspection PyTypeChecker
    return round(average_cell_height / P_1_DIVISION_FACTOR)


def get_imagedata(image_filename: str) -> dict:
    img = cv2.imread(image_filename)

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
    # image_as_dataframe: DataFrame = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)

    return imagedata


if __name__ == "__main__":
    logger.info("Started processing on: " + os.path.basename(IMAGE_FILE))
    pytesseract_imagedata = get_imagedata(IMAGE_FILE)

    P_1_HORIZONTAL_WORD_SPACING_DISTANCE = determine_horizontal_spacing_distance(pytesseract_imagedata)
    logger.info("Determined the P_1_HORIZONTAL_WORD_SPACING_DISTANCE to [" + str(
        P_1_HORIZONTAL_WORD_SPACING_DISTANCE) + "] pts")
    P_3_VERTICAL_LINE_SPACING_DISTANCE = determine_vertical_spacing_distance(pytesseract_imagedata)
    logger.info(
        "Determined the P_3_VERTICAL_LINE_SPACING_DISTANCE to [" + str(P_3_VERTICAL_LINE_SPACING_DISTANCE) + "] pts")

    data_recognition(pytesseract_imagedata, img=cv2.imread(IMAGE_FILE))

    # todo 2.3 cell recovery
