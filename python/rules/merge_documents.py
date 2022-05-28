import argparse
import glob
import json
import os.path
from typing import List

from loguru import logger

from docrecjson.elements import Document, ContentObject, PolygonRegion
from docrecjson import decoder

import evaluations.iou as iou

from shapely.geometry import Polygon


def _load_document(filepath: str) -> Document: # noqa removing this warning because dup originates from submodule
    with open(filepath) as json_data:
        json_annotation = json.load(json_data)

    return decoder.loads(json.dumps(json_annotation))


def get_prediction(filename: str, prediction_dir: str) -> Document:
    filename = filename.replace(".png", "")
    filename_without_extension = os.path.basename(os.path.splitext(filename)[0])
    glob_searchstring: str = os.path.join(prediction_dir, filename_without_extension) + ".*"
    matching_gt = []
    for filepath in glob.glob(glob_searchstring):
        matching_gt.append(filepath)

    if len(matching_gt) != 1:
        raise RuntimeError(
            "Expected number of matching annotation file for image file to be 1, actual number was: " + str(
                len(matching_gt)))
    else:
        logger.info("Found matching annotation file for [" + filename + "]: [" + matching_gt[0] + "]")

    gt_annotation_path: str = matching_gt[0]
    return _load_document(gt_annotation_path)


def write_to_file(filepath: str, document: Document):
    dct: dict = document.to_dict()
    if filepath is not None:
        with open(filepath, "w") as file:
            json.dump(dct, file, indent=2)
            logger.info("wrote processed contents into: [" + filepath + "]")


def get_intersecting_cells(content_object, polygon_content_d2) \
        -> List[PolygonRegion]:
    p_1_area: Polygon = Polygon(content_object.polygon)
    intersecting_cells: list = []
    for element in polygon_content_d2:
        p_2_area: Polygon = Polygon(element.polygon)
        if p_1_area.intersects(p_2_area):
            intersecting_cells.append(element)
    return intersecting_cells


def merge_bounding_boxes():
    pass


def merge_documents(document_1: Document, document_2: Document) -> Document:
    if document_1.filename != document_2.filename:
        raise RuntimeError("No matching filename between two document versions.")
    if document_1.original_image_size != document_2.original_image_size:
        raise RuntimeError("No matching image size between the two document versions.")

    result: Document = Document.empty(filename=document_1.filename, original_image_size=document_1.original_image_size)
    result.add_existing_metadata(document_1.meta)
    result.add_existing_metadata(document_2.meta)

    result.add_metadata({"creator": "merge-documents"})

    polygon_content_d1 = [x for x in document_1.content if type(x) == PolygonRegion]
    polygon_content_d2 = [x for x in document_2.content if type(x) == PolygonRegion]

    prediction_2_viewed_oid: list = []

    for content_object in polygon_content_d1:
        intersecting_cells = get_intersecting_cells(content_object, polygon_content_d2) # noqa we filter this type previoulsy

        if len(intersecting_cells) == 0:
            result.add_region(area=content_object.polygon, region_type="text")
        elif len(intersecting_cells) == 1:
            result.add_region(area=content_object.polygon, region_type="text")
            prediction_2_viewed_oid.append(intersecting_cells[0].oid)
        else:
            result.add_region(area=content_object.polygon, region_type="text")
            max_iou_element: PolygonRegion
            max_iou_value: float = 0
            for intersecting_cell in intersecting_cells:
                intersection_over_union = iou.cell_intersection_over_union(content_object, intersecting_cell) # noqa we know that these are PolygonRegions because we filtered the polygon reginos already
                if intersection_over_union > max_iou_value:
                    max_iou_element = intersecting_cell

            prediction_2_viewed_oid.append(max_iou_element.oid)

    for content_object in polygon_content_d2:
        if content_object.oid not in prediction_2_viewed_oid:
            result.add_region(area=content_object.polygon, region_type="text")

    return result


def main(prediction_dir_1: str, prediction_dir_2: str, output_dir: str):
    for filepath in glob.glob(os.path.join(prediction_dir_1, "*")):
        filename = os.path.basename(filepath)

        logger.info("Receiving file [" + str(filepath) + "]")

        prediction_1: Document = _load_document(filepath)
        prediction_2: Document = get_prediction(filename, prediction_dir_2)

        document: Document = merge_documents(prediction_1, prediction_2)

        write_to_file(os.path.join(output_dir, filename), document)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--prediction_dir_1", type=str, required=True, help="The first prediction directory.")
    parser.add_argument("-p2", "--prediction_dir_2", type=str, required=True, help="The second prediction directory.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="The directory where the merged files are written into.")

    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args.prediction_dir_1, args.prediction_dir_2, args.output_dir)
