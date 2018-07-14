#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import dlib
from PIL import Image
from sklearn.externals import joblib
from skimage import io
from skimage.color import rgb2grey
from skimage.transform import warp, rescale, SimilarityTransform

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def register_face(face, detector, predictor):
    frame = detector(face, 1)
    if len(frame) == 1:
        shape = predictor(face, frame[0])
        eye1 = np.mean(np.array([[shape.part(i).x,
                                  shape.part(i).y] for i in range(0,2)]),0)
        eye2 = np.mean(np.array([[shape.part(i).x,
                                 shape.part(i).y] for i in range(2,4)]),0)
       
        ###Rotate
        angle = -np.arctan2(eye1[1]-eye2[1], eye1[0]-eye2[0])
        tform = SimilarityTransform(rotation=-angle)
        eye1 = np.array([eye1[0]*np.cos(angle)-eye1[1]*np.sin(angle),
                         eye1[0]*np.sin(angle)+eye1[1]*np.cos(angle)])
    
        eye2 = np.array([eye2[0]*np.cos(angle)-eye2[1]*np.sin(angle),
                         eye2[0]*np.sin(angle)+eye2[1]*np.cos(angle)])
        face = warp(face, tform)
        
        ###Scale
        scale = 50/(eye1[0] - eye2[0])
        eye1 = eye1*scale
        eye2 = eye2*scale
        face = rescale(face, scale, mode='reflect')
        
        ###Translate
        tform = SimilarityTransform(translation=(eye2[0]-25,eye2[1]-25))
        face = warp(face, tform)
        
        ###Crop
        face = face[:100,:100]
        
        return 255*face.astype(np.float32)
    
    return("Unable to detect face")

##Create a face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face = register_face(face,detector,predictor)