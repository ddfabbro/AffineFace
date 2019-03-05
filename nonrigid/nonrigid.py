import numpy as np
from skimage import transform, draw
from scipy.spatial import Delaunay
from joblib import Parallel, delayed

def nonrigid(img, landmarks, template, n_jobs=-1):
    def warp_triangles(src, dst):
        X, Y = draw.polygon(dst[:,0], dst[:,1])
        tform = transform.estimate_transform("affine", dst, src)
        warped_img[Y, X] = transform.warp(img, tform)[Y, X]
        
    simplices = Delaunay(template).simplices
    template_vertices = template[simplices]
    img_vertices = landmarks[simplices]
    
    warped_img = np.zeros(img.shape)
    Parallel(n_jobs=n_jobs, require="sharedmem")(delayed(warp_triangles)
        (src, dst) for src, dst in zip(img_vertices, template_vertices))
        
    return warped_img