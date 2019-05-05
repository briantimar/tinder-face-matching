import openface
import numpy as np
import os
import itertools
import cv2
import argparse
import openface  
import face_utils

# !/usr/bin/env python2
#
# Based on the compare.py demo in Openface by Brandon Aomos

import time

start = time.time()

#path to openface root
openfaceRootDocker = "/root/openface"
openfaceRoot = openfaceRootDocker

# the .lua torch models
modelDir = os.path.join(openfaceRoot, 'models')
# the dlib models
dlibModelDir = os.path.join(modelDir, 'dlib')
# the python interface to the torch models
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

# load the dlib and torch models
start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))

bgrIm = face_utils.getBGRArray(args.imgs[0])
rgbIm = cv2.cvtColor(bgrIm, cv2.COLOR_BGR2RGB)
boxes_and_reps = face_utils.pull_boxes_and_reps(rgbIm, align, net)
# faces, boxes= face_utils.get_all_aligned_faces(bgrIm, align)
# cv2.imwrite("annotated_0.png", faces[0])
