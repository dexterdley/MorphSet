#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:28:29 2020

@author: dexter
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.nn.functional as F
import pdb
import torch.nn as nn
import cv2

import torch
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()

def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    x = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]].shape[1]
    y = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]].shape[0]
    
    # Get mask by filling triangle
    mask = np.zeros((y, x, 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    #size = (r[2], r[3])
    size = (x, y)
    
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def get_landmarks(img, detector, predictor):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(img_gray)
     
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        
        landmarks_points = []
        for n in range(0, 68): #get the landmarks of face
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            
            #cv2.circle(img, (x,y), 3, (0,0,255), -1)
            
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points) #create outline from points
    
    cv2.fillConvexPoly(mask, convexhull, 255)
    # Display the image.
    face_image = cv2.bitwise_and(img, img, mask=mask)
    
    #return face_image, landmarks_points
    return img, landmarks_points

def delaunay_triangulation(points):
    convexhull = cv2.convexHull(points) #create outline from points
  
    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)

    subdiv.insert(points.tolist()) #points need to be in list type
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
    
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
    
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
    
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
    
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    
    return indexes_triangles
    

def compute_CCC(y_true, y_pred):
    '''
    Parameters
    ----------
    y_true : tensor
        Ground truth vector
    y_pred : tensor
        Precicted values

    Returns
    -------
    CCC, a float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.

    Reference code: https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
    '''
    cor = np.corrcoef(y_true,y_pred)[0,1] #return correlation covariance matrix, take [0][1] as pearson coefficient
    
    mean_true, var_true, sd_true = y_true.mean(), y_true.var(), y_true.std()
    mean_pred, var_pred, sd_pred = y_pred.mean(), y_pred.var(), y_pred.std()
 
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator, cor 

# Check accuracy on training to see how good our model is
def check_scores(loader, model):
    """
    Parameters
    ----------
    loader : Dataloader
        Validation dataset
    model : TYPE
        DESCRIPTION.
    
    Using only 3-channel RGB model
    
    Returns
    -------
    RMSE and CCC values for both Arousal and Valence predictions
    """
    
    num_samples = 0
    model.eval()
    cpu = torch.device('cpu')

    with torch.no_grad():
        
        error_V, error_A = 0.0, 0.0
        CCC_V = []
        CCC_A = []
        corr_V = []
        corr_A = []
        
        test_losses = []
        
        for count, (x, y) in enumerate(loader):
            
            x = x.to(device = device)
            y = y.to(device = device)
            
            #forward
            score = model(x)
                
            valence_pred = score[:,0]
            arousal_pred = score[:,1]
            
            loss = criterion(score, y)
            test_losses.append(loss.item())
            
            error_V += torch.sum( torch.square(valence_pred - y[:, 0]) )
            error_A += torch.sum( torch.square(arousal_pred - y[:, 1]) )
            
            ccc_v, corr_v = compute_CCC( y[:, 0].to(cpu), valence_pred.to(cpu))
            ccc_a, corr_a = compute_CCC( y[:, 1].to(cpu), arousal_pred.to(cpu))
             
            CCC_V.append(ccc_v)
            CCC_A.append(ccc_a)
            corr_V.append(corr_v)
            corr_A.append(corr_a)
            
            #pdb.set_trace()    
            
            num_samples += len(x)
        
        #print(num_samples)
        
        error_V = error_V.to(cpu)
        error_A = error_A.to(cpu)
        
        mse_V = error_V/ num_samples
        mse_A = error_A/ num_samples
        
        CCC_Valence = np.mean(CCC_V)
        CCC_Arousal = np.mean(CCC_A)
        
        Corr_V = np.mean(corr_V)
        Corr_A = np.mean(corr_A)
        
        return sum(test_losses)/len(test_losses), np.sqrt(mse_V), np.sqrt(mse_A), CCC_Valence, CCC_Arousal, Corr_V, Corr_A
        
# Check accuracy on training to see how good our model is
def check_scores_landmarks(loader, model):
    
    model.eval()
    cpu = torch.device('cpu')
    
    with torch.no_grad():
        
        error_V, error_A = 0.0, 0.0
        CCC_V = []
        CCC_A = []
        corr_V = []
        corr_A = []
        
        for count, (x, y, landmarks) in enumerate(loader):
            
            x = x.to(device = device)
            y = y.to(device = device)
            landmarks = landmarks.to(device = device)
            
            #forward
            score = model(x, landmarks)
                
            valence_pred = score[:,0]
            arousal_pred = score[:,1]
                
            
            error_V += torch.sum( torch.square(valence_pred - y[:, 0]) )
            error_A += torch.sum( torch.square(arousal_pred - y[:, 1]) )
            
            ccc_v, corr_v = compute_CCC( y[:, 0].to(cpu), valence_pred.to(cpu))
            ccc_a, corr_a = compute_CCC( y[:, 1].to(cpu), arousal_pred.to(cpu))
             
            CCC_V.append(ccc_v)
            CCC_A.append(ccc_a)
            corr_V.append(corr_v)
            corr_A.append(corr_a)
            
            #pdb.set_trace()    
        
        error_V = error_V.to(cpu)
        error_A = error_A.to(cpu)
        
        mse_V = error_V/ 4500
        mse_A = error_A/ 4500
        
        CCC_Valence = np.mean(CCC_V)
        CCC_Arousal = np.mean(CCC_A)
        
        Corr_V = np.mean(corr_V)
        Corr_A = np.mean(corr_A)
        
        return np.sqrt(mse_V), np.sqrt(mse_A), CCC_Valence, CCC_Arousal, Corr_V, Corr_A
  
def save_checkpoint(state, epoch, expt_name):
    print("==> Checkpoint saved")
    
    if not os.path.exists('./models/' + expt_name):
        os.makedirs('./models/' + expt_name)
        
    outfile = './models/' + expt_name + '/' + str(epoch) + '_' + expt_name + '.pth.tar'
    torch.save(state, outfile)
    
def load_checkpoint(model, optimizer, weight_file):
    print("==> Loading Checkpoint: " + weight_file)
    
    #weight_file = r'checkpoint.pth.tar'
    if torch.cuda.is_available() == False:
        checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(weight_file)
        
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
def onehot(pred):
    onehot = []
    for item in pred:
        y_vec = np.zeros(3)
        y_vec[item] = 1
        onehot.append(y_vec)
        
    return torch.FloatTensor(onehot)