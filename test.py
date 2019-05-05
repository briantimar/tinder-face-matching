import openface
import numpy as np
import glob
import os
from face_utils import pull_boxes_and_reps, dist
from face_utils import get_bgr_img
from face_utils import get_align_and_net, bgr_to_rgb

target_dir = "test_images/openface_test"
images = glob.glob( os.path.join(target_dir, "*.jpg"))

print("Found %d images" % len(images))

bgr_images = [get_bgr_img(impath) for impath in images]
rgb_images = [ bgr_to_rgb(im) for im in bgr_images]

align, net = get_align_and_net()
boxes_and_reps = list(map(lambda im: pull_boxes_and_reps(im,align,net), 
                                    rgb_images))
clapton_fname = filter(lambda s: "clapton" in s, images)[0]
clapton = boxes_and_reps[ images.index(clapton_fname)][0]['rep']

for i in range(len(images)):
    br = boxes_and_reps[i]
    for face in br:
        print("name {0}, distance from clapton {1:.3f}".format(images[i], 
                                                        dist(face['rep'], clapton)))



