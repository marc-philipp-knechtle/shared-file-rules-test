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
