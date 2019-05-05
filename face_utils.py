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




# def getRep(imgPath):
#     """Return representation of face in the specified path"""
#     if args.verbose:
#         print("Processing {}.".format(imgPath))
#     #loads image from path into numpy uint array
#     bgrImg = cv2.imread(imgPath)
#     if bgrImg is None:
#         raise Exception("Unable to load image: {}".format(imgPath))
#     #flips the channel order (I guess this is necessary by default?)
#     rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

#     if args.verbose:
#         print("  + Original size: {}".format(rgbImg.shape))

#     start = time.time()
#     # a dlib rectangle object bounding the face
#     bb = align.getLargestFaceBoundingBox(rgbImg)
#     if bb is None:
#         raise Exception("Unable to find a face: {}".format(imgPath))
#     if args.verbose:
#         print("  + Face detection took {} seconds.".format(time.time() - start))

#     start = time.time()
#     #numpy array holding the aligned face
#     alignedFace = align.align(args.imgDim, rgbImg, bb,
#                               landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
#     if alignedFace is None:
#         raise Exception("Unable to align image: {}".format(imgPath))
#     if args.verbose:
#         print("  + Face alignment took {} seconds.".format(time.time() - start))

#     start = time.time()
#     rep = net.forward(alignedFace)
#     if args.verbose:
#         print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))

#     return rep, bb, alignedFace
