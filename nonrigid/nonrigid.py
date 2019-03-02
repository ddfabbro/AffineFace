import numpy as np
import skimage.transform as tf
from skimage.draw import polygon
from scipy.spatial import Delaunay

def nonrigid(img, landmarks, template):
    simplices = Delaunay(template).simplices
    template_vertices = template[simplices]
    img_vertices = landmarks[simplices]
    
    warped_img = np.zeros(img.shape)
    for src, dst in zip(img_vertices, template_vertices):
        X, Y = polygon(dst[:,0], dst[:,1])
        tform = tf.estimate_transform('affine', dst, src)
        warped_img[Y, X] = tf.warp(img, tform)[Y, X]
        
    return warped_img