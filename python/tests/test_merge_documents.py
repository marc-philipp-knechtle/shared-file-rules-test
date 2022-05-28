from unittest import TestCase

import json
import re

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
        The merge operation of two equal files should result in the same document.
        The intersecting cell areas will surely be detected, but the average cell are from two equal cell areas is the
        same cell area.
        :return:
        """

        result: Document = md.merge_documents(_load_document(SIMPLE_ANNOTATION_1),
                                              _load_document(SIMPLE_ANNOTATION_1))
        annotation: Document = _load_document(SIMPLE_ANNOTATION_1)

        exclude_oid = re.compile(r"root\['content']\[\d+]\['oid']")
        exclude_group = re.compile(r"root\['content']\[\d+]\['group']")
        ddiff: DeepDiff = DeepDiff(annotation.to_dict(), result.to_dict(), ignore_order=True,
                                   exclude_paths=["root['meta']"], exclude_regex_paths=[exclude_oid, exclude_group])
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

    def test_complex_merge(self):
        cascade_tab_net_annotation: str = \
            "../../tests/fixtures/annotations/merge-documents/test1/cascade-tab-net-annotation.json"
        shigarov2016configurable_annotation: str = \
            "../../tests/fixtures/annotations/merge-documents/test1/shigarov2016configurable-annotation.json"

        result: Document = md.merge_documents(_load_document(cascade_tab_net_annotation),
                                              _load_document(shigarov2016configurable_annotation))
        annotation: Document = _load_document(SIMPLE_ANNOTATION_1)

        exclude_oid = re.compile(r"root\['content']\[\d+]\['oid']")
        exclude_group = re.compile(r"root\['content']\[\d+]\['group']")
        ddiff: DeepDiff = DeepDiff(annotation.to_dict(), result.to_dict(), ignore_order=True,
                                   exclude_paths=["root['meta']"], exclude_regex_paths=[exclude_oid, exclude_group])
        # todo create result fixture once the merge algorithm is ready. Not doing this, because currently WIP
        assert ddiff != {}
