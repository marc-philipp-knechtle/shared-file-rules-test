from unittest import TestCase

import json

from docrecjson import decoder
from docrecjson.elements import Document

import python.rules.merge_documents as md

from deepdiff import DeepDiff

SIMPLE_ANNOTATION_1: str = "../../tests/fixtures/annotations/merge-documents/simple-annotation-1.json"


def _load_document(filepath: str) -> Document:
    with open(filepath) as json_data:
        json_annotation = json.load(json_data)

    return decoder.loads(json.dumps(json_annotation))


class Test(TestCase):
    def test_merge_equal_files(self):
        """
        The merge operation of two equal files should result in the same document
        :return:
        """

        result: Document = md.merge_documents(_load_document(SIMPLE_ANNOTATION_1),
                                              _load_document(SIMPLE_ANNOTATION_1))
        annotation: Document = _load_document(SIMPLE_ANNOTATION_1)
        # todo deepdiff test issues with other id's -> exclude metadata + id values from this computation
        ddiff: DeepDiff = DeepDiff(annotation.to_dict(), result.to_dict(), ignore_order=True)
        assert ddiff == {}

    def test_addition_merge(self):
        """
        Only cells with no common ground are merged. This results in a document with cells from both document versions.
        """
        pass

    def test_iou_merge(self):
        """
        The IoU rule applies here. There are multiple candidates for a cell merge. We take the candidate with the highest
        Intersection over Union.
        """
        pass
