import openface
import numpy as np
import glob
import os
import cv2
from face_utils import pull_boxes_and_reps, dist
from face_utils import get_bgr_img
from face_utils import get_align_and_net, bgr_to_rgb
from face_utils import locate_common_reps
from face_utils import annotate_image

def test_openface_targets():
    target_dir = "test_images/openface_test"
    images = glob.glob( os.path.join(target_dir, "*.jpg"))

    print("Found %d images" % len(images))

    bgr_images = [get_bgr_img(impath) for impath in images]
    rgb_images = [ bgr_to_rgb(im) for im in bgr_images]

    align, net = get_align_and_net()
    boxes_and_reps = list(map(lambda im: pull_boxes_and_reps(im,align,net), 
                                        rgb_images))
    clapton_fname = list(filter(lambda s: "clapton" in s, images))[0]
    clapton = boxes_and_reps[ images.index(clapton_fname)][0]['rep']

    for i in range(len(images)):
        br = boxes_and_reps[i]
        for face in br:
            print("name {0}, distance from clapton {1:.3f}".format(images[i], 
                                                            dist(face['rep'], clapton)))

def test_local_targets():
    target_dir = "test_images/user1"
    images = glob.glob(os.path.join(target_dir, "*.jpg"))

    print("Found %d images" % len(images))
    print(images)

    bgr_images = [get_bgr_img(impath) for impath in images]
    rgb_images = [bgr_to_rgb(im) for im in bgr_images]
    align, net = get_align_and_net()
    boxes_and_reps = list(map(lambda im: pull_boxes_and_reps(im, align, net),
                              rgb_images))

    reps = [[face['rep'] for face in im] for im in boxes_and_reps]

    common_rep_indices = locate_common_reps(reps, keep_all_singles=True)
    print("User indices:", common_rep_indices)
    for i in range(len(images)):
        if common_rep_indices[i] is None:
            print("No user found in image %d"%i)
            labeled_im = bgr_images[i]
        else:
            user_face = boxes_and_reps[i][common_rep_indices[i]]
            labeled_im = annotate_image(bgr_images[i], user_face['box'])
        cv2.imwrite(images[i].split('.')[0] + '_labeled.jpg', labeled_im)
    
test_local_targets()

