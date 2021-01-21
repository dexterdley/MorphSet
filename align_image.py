# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:20:08 2020

@author: Dex
"""
import numpy as np
import scipy.ndimage
import cv2

FACIAL_LANDMARKS_IDXS = dict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("left_eye", (36, 42)),
	("right_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

# hardcoded specification values for cropping around the eyes
ex = 0.7 #x cropping factor to the left and right of the eyes
eyup = 1.2 #y cropping factor to the upper of the eyes
eydown = 1.8 #y cropping factor to the bottom of the eyes


def align_image(points, img, e2e):
    
    '''
    Inputs:
        points (landmark points)
        img (Image)
        e2e (eye to eye distance)
        
    Outputs:
        aligned and resized 450x360x3 image
    '''
    
    
    leftEyepts = points[ FACIAL_LANDMARKS_IDXS["left_eye"][0]: FACIAL_LANDMARKS_IDXS["left_eye"][1] ]
    rightEyepts = points[ FACIAL_LANDMARKS_IDXS["right_eye"][0]: FACIAL_LANDMARKS_IDXS["right_eye"][1] ]

        
    left_ctr = tuple( np.mean(leftEyepts, axis=0).astype(int) )
    right_ctr = tuple( np.mean(rightEyepts, axis=0).astype(int) )

    points = np.vstack([points, left_ctr])
    points = np.vstack([points, right_ctr])

    minxy = np.min(points, axis=0)
    points[:,0] = points[:,0] - minxy[0]
    points[:,1] = points[:,1] - minxy[1]


    dx = points[69,0] - points[68,0]
    dy = points[69,1] - points[68,1]
    dist = np.sqrt( np.square(dx) + np.square(dy)) #find dist between eye centres using pythagoras

    if dx != 0:
        f = dy/dx
        a =  np.arctan(f)
    
    R = np.ones([2,2]) # construct rotation matrix
    R[0,0] = np.cos(a)
    R[0,1] = -np.sin(a)
    R[1,0] = np.sin(a)
    R[1,1] = np.cos(a)

    points = points @ R # apply rotation matrix

    minxy = np.min(points, axis=0)
    points[:,0] = points[:,0] - minxy[0]
    points[:,1] = points[:,1] - minxy[1]

    scale_factor = e2e/dist
    points = points * scale_factor

    # cropping arroud the eyes
    x1 = points[68,0] - ex * e2e
    x2 = points[69,0] + ex * e2e
    y1 = points[68,1] - eyup * e2e
    y2 = points[69,1] + eydown * e2e
    x1 = np.floor(x1)
    y1 = np.floor(y1)
    x2 = np.floor(x2)
    y2 = np.floor(y2)

    Npoints = np.ones(points.shape)
    #adjusting coordinates to the new croppings: new points
    Npoints[:, 0] = points[:, 0] - x1
    Npoints[:, 1] = points[:, 1] - y1


    # expressing eyes according to the image center, because this is
    # the only part that remains the same during rotation
    c = np.array([1,2])
    c[0] = img.shape[1] /2  #center of the image x
    c[1] = img.shape[0] /2 #center of the image y

    left_ctr = left_ctr - c
    right_ctr = right_ctr - c 

    # rotating eyes
    left_ctr = left_ctr @ R
    right_ctr = right_ctr @ R

    img = scipy.ndimage.rotate(img, a)
    c[0] = img.shape[1] /2  #center of the rotated image x
    c[1] = img.shape[0] /2 #center of the rotated image y 

    left_ctr = left_ctr + c
    right_ctr = right_ctr + c 

    # scaling eyes
    scale = (int(img.shape[1] *scale_factor) , int(img.shape[0] *scale_factor))

    img = cv2.resize(img, scale)

    left_ctr = (left_ctr * scale_factor).astype(int)
    right_ctr = (right_ctr * scale_factor).astype(int)

    # cropping around the eyes according to the specifications
    x1 = left_ctr[0] - ex * e2e
    x2 = right_ctr[0] + ex * e2e
    y1 = left_ctr[1] - eyup * e2e
    y2 = left_ctr[1] + eydown * e2e
    x1 = np.floor(x1).astype(int)
    y1 = np.floor(y1).astype(int)
    x2 = np.floor(x2).astype(int)
    y2 = np.floor(y2).astype(int)

    croppedI = img[y1:y2, x1:x2]
    Npoints = np.array(Npoints, np.int32)
    
    assert croppedI.shape[0] >= 400 and croppedI.shape[1] >= 300
    
    return croppedI, Npoints
    
    
    