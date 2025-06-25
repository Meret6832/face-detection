import cv2
import dlib
import logging
import math
import numpy as np
import os
import pandas as pd
import sys
import time

MAX_PIXELS = 1e7
# Face angle
ANGLE_THRESH = 35
# Cropping
ASPECT_RATIO = 35/45 # Width / height
MAX_MARGIN_W = 0.4 # Max fraction of width of face that can be around detected face bb.
MAX_MARGIN_H = MAX_MARGIN_W / ASPECT_RATIO
# Lighting detection
CONTRAST_THRESH = 0.05  # Set this lower for stricter low contrast detection.
EXPOSURE_THRESH = 0.8  # Set this lower for stricter under-/over-exposure detection.
# Blur detection
BLUR_THRESH = 15  # Set this higher for stricter blur detection.

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def check_landmarks(grey_img, face_rect, show_landmarks=False):
    # source: https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    face_shape = predictor(grey_img, face_rect)
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (face_shape.part(i).x, face_shape.part(i).y)

    # Check angle between eyes (most right point of right eye <--> most left point of left eye)
    dx = coords[45, 0] - coords[36, 0]
    dy = coords[45, 1] - coords[36, 1]
    angle = math.atan(dy/dx) * 180 / math.pi
    if abs(angle) > ANGLE_THRESH:
        logger.info("Face is too tilted, angle={:.1f}".format(angle))
        return False

    nose = coords[27:36]
    if show_landmarks:
        for (x, y) in coords:
	        cv2.circle(grey_img, (x, y), 1, (0, 0, 255), 3)
        cv2.imshow("out", grey_img)
        cv2.waitKey(0)

    # Check if nose is between eyes
    if coords[45,0] < max(nose[:,0]) or coords[36,0] > min(nose[:,0]):
        logger.info("Nose is not between eyes")
        return False
    
    # Check if eyes are in between jaw
    if coords[0,0] > min(nose[:,0]) or coords[16,0] < max(nose[:,0]):
        logger.info("Eyes are not in between jaw")
        return False
    
    return True


def center_face(img, rect):
    img_h, img_w, _ = img.shape
    rect_w = rect.right() - rect.left()
    rect_h = rect.bottom() - rect.top()

    left_margin = rect.left() / rect_w
    right_margin = (img_w - rect.right()) / rect_w
    top_margin = rect.top() / rect_h
    bottom_margin = (img_h - rect.bottom()) / rect_h

    if (left_margin > MAX_MARGIN_W or right_margin > MAX_MARGIN_W or
        top_margin > MAX_MARGIN_H or bottom_margin > MAX_MARGIN_H):
        print("Centering face...")

        target_w = min(round(rect_w + 2*MAX_MARGIN_W*rect_w), img_w)
        target_h = round(target_w / ASPECT_RATIO)
        if target_h > img_h:
            target_h = img_h
            target_w = round(target_h * ASPECT_RATIO)

        new_left = round(max(min(img_w-target_w, rect.center().x - target_w/2), 0))
        new_right = new_left + target_w
        new_top = round(max(min(img_h-target_h, rect.center().y - target_h/2), 0))
        new_bottom = new_top + target_h
        img = img[new_top:new_bottom, new_left:new_right]
        return (img, [new_top, new_bottom, new_left, new_right])
    
    return (img, [])


def check_lighting(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([grey_img], [0], None, [256], [0, 256])
    hist_frac = hist / hist.sum()
    dark_frac = np.sum(hist_frac[0:50])
    light_frac = np.sum(hist_frac[-50:])
    # print("dark", dark_frac, "light", light_frac)
    if (dark_frac > EXPOSURE_THRESH):
        logger.info("Image is underexposed")
        return False
    if (light_frac > EXPOSURE_THRESH):
        logger.info("Image is overexposed")
        return False
    
    return True

def check_blur(img):
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()  # variance of convolution with Laplacian kernel.

    if lap_var < BLUR_THRESH:
        logger.info(f"Image is blurry, Laplacian variation={lap_var}")
        return False
    
    return True

def calibrate_blur():
    with open("blur_thresh.csv", "w+") as f:
        f.write("name,blur,var\n")

    for file in os.listdir("../examples/blur"):
        print(file)
        img = cv2.imread(f"../examples/blur/{file}")
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rects = detector(grey_img, 1)
        if len(face_rects) != 1: continue
        img, _ = center_face(img, face_rects[0])
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        with open("blur_thresh.csv", "a") as f:
            f.write(f"{file},blur,{var}\n")

    for file in os.listdir("../examples/good"):
        print(file)
        img = cv2.imread(f"../examples/good/{file}")
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rects = detector(grey_img, 1)
        if len(face_rects) != 1: continue
        img, _ = center_face(img, face_rects[0])
        with open("blur_thresh.csv", "a") as f:
            f.write(f"{file},not,{var}\n")

    df = pd.read_csv("blur_thresh.csv")
    print("\nblur\tmean\tmedian\tmin\tmax")
    for b in ["blur", "not"]:
        x = df[df["blur"] == b]
        print("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(b, x['var'].mean(), x['var'].median(), x['var'].min(), x['var'].max()))


def validate_img(img_path, reduce_quality=True, show_landmarks=False, show_crop=False):
    # check_point = time.time()
    img = cv2.imread(img_path)
    x, y, _ = img.shape
    if reduce_quality and x*y > MAX_PIXELS:
        scale_factor = MAX_PIXELS/(x*y)
        img = cv2.resize(img, (round(x*scale_factor), round(y*scale_factor)))
        x, y, _ = img.shape
        assert x*y <= MAX_PIXELS

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rects = detector(grey_img, 1) # second parameter: num of image pyramid layers. Higher means detecting more faces but also more computationally expensive

    # print("loading image: ", time.time()-check_point)
    # check_point = time.time()    

    if len(face_rects) > 1:
        logger.info(f"Found {len(face_rects)} faces.")
        return (False, [])
    elif len(face_rects) == 0:
        logger.info("Found no faces.")
        return (False, [])

    # Center face, assuming picture has correct aspect ratio to begin with.
    cropped_img, crop_coords = center_face(img, face_rects[0])
    if show_crop:
        top, bottom, left, right = crop_coords
        cv2.rectangle(img, (left,top), (right,bottom), (0,0,255), 3)
        cv2.rectangle(img, (face_rect.left(),face_rect.top()), (face_rect.right(),face_rect.bottom()), (0,255,0), 3)
        cv2.imshow("out", img)
        cv2.waitKey(0)
    
    # print("cropping: ", time.time()-check_point)
    # check_point = time.time() 

    if not check_lighting(cropped_img): 
        return (False, [])
    if not check_blur(cropped_img):
        return (False, [])
    
    # print("checking lighting/blur: ", time.time()-check_point)
    # check_point = time.time() 

    # Check face rotation with facial landmarks
    face_rects = detector(grey_img, 1)
    if len(face_rects) > 1:
        print(f"Found {len(face_rects)} faces.")
        return (False, [])
    elif len(face_rects) == 0:
        print("Found no faces.")
        return (False, [])
    face_rect = face_rects[0]
    
    cropped_img_grey = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    if not check_landmarks(cropped_img_grey, face_rect, show_landmarks):
        return (False, [])
    
    # print("checking rotation face: ", time.time()-check_point)
    # check_point = time.time() 

    return (True, crop_coords)

def validate(reduce=True):
    if reduce:
        csv_path = "validate_reduce.csv"
    else:
        csv_path = "validate.csv"
    print("Writing validation to", csv_path)

    with open(csv_path, "w+") as f:
        f.write("name,cat,res,correct,t,n_pixels\n")

    # calibrate_blur()
    for folder in os.listdir("validation"):
        correct_res = False
        if folder == "accept":
            correct_res = True
        for file in os.listdir(f"validation/{folder}"):
            start = time.time()
            res, _ = validate_img(f"validation/{folder}/{file}", reduce_quality=reduce)
            run_time = time.time() - start

            img = cv2.imread(f"validation/{folder}/{file}")
            x, y, _ = img.shape

            with open(csv_path, "a") as f:
                f.write(f"{file},{folder},{res},{res==correct_res},{run_time},{x*y}\n")


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    if len(sys.argv) > 2:
        reduce = bool(sys.argv[2])
    
    print("reduce is set to", reduce)

    if sys.argv[1].strip().lower() == "validate":
        print("Validating...")
        validate(reduce=reduce)
    else:
        res = validate_img(sys.argv[1], reduce_quality=reduce, show_landmarks=False, show_crop=False)
