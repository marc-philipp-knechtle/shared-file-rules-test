import json

from docrecjson import decoder
from docrecjson.elements import Document

import python.rules.merge_documents as merge_documents

SIMPLE_ANNOTATION_1: str = "fixtures/annotations"


def _load_document(filepath: str) -> Document:
    with open(filepath) as json_data:
        json_annotation = json.load(json_data)

    return decoder.loads(json.dumps(json_annotation))


def test_merge_equal_files():
    """
    The merge operation of two equal files should result in the same document
    :return:
    """
    pass


def test_addition_merge():
    """
    Only cells with no common ground are merged. This results in a document with cells from both document versions.
    """
    pass


def test_iou_merge():
    """
    The IoU rule applies here. There are multiple candidates for a cell merge. We take the candidate with the highest
    Intersection over Union.
    """
    pass
