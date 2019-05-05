""" Utilities for finding and annotating faces.
    Taken from openface compare demo and annotation util."""

import openface
import numpy 
import cv2

def getBGRArray(imgPath):
    """Return numpy array corresponding to the image at the specified path, with channels in 
    BGR order."""
    return cv2.imread(imgPath)

def get_bounding_boxes(image, align):
    """ Return all face bounding boxes in a given image.
        image: a numpy array
        align: an alignDlib object.
        returns: list of dlib rectangles."""
    return align.getAllFaceBoundingBoxes(image)

def annotate_image(image, bounding_boxes):
    """Write the bounding boxes provided onto the image.
        image = a numpy array
        bounding_boxes: an iterable holding dlib rectangles.
        returns: new image with each bounding box marked.
        """
    image = image.copy()
    for box in bounding_boxes:
        bl = (box.left(), box.bottom())
        tr = (box.right(), box.top())
        cv2.rectangle(image, bl, tr, color=(153, 255, 204), thickness=5)
    return image

def get_all_aligned_faces(image, align, imgDim=96):
    """ Returns all aligned faces found in the image at the specified path.
        image: imagearray
        align: aligndlib object.
        imgDim: int, the linear size of the aligned images. Default: 96
        Returns: list of aligned faces, and list of corresponding bounding boxes.
        """

    bboxes = get_bounding_boxes(image, align)
    if len(bboxes) ==0:
        raise ValueError("No faces found!")
    aligned_faces = []
    for box in bboxes:
        alignedFace = align.align(imgDim, image, box,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        aligned_faces.append(alignedFace)
    return aligned_faces, bboxes

def pull_boxes_and_reps(image, align, net, imgDim=96):
    """ Given an RGB image, return bounding boxes and reps for all faces.
    image: numpy image in RGB format.
    align: an aligndlib object
    imgDim: linear image size to use for aligned images.
    net: openface interface to the torch neural net

    Returns: a list, holding bounding_box, rep pairs in dictionaries 
    for each face found.
    """

    newfaces = []
    aligned_faces, bboxes = get_all_aligned_faces(image, align, imgDim=imgDim)
    for i in range(len(aligned_faces)):
        newfaces.append(dict(box=bboxes[i],
                            rep=net.forward(aligned_faces[i])))
    return newfaces

