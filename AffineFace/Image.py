from __future__ import division
import PIL
import numpy as np
from numpy.linalg import norm, inv

class Image(object):
    def __init__(self, fp, *args, **kwargs):
        self.im = PIL.Image.open(fp)

    def align(self, eyes, **kwargs):
        size = kwargs.pop('size', self.im.size)
        eyes_distance = kwargs.pop('eyes_distance', norm(eyes[1] - eyes[0]))
        x_offset = kwargs.pop('x_offset', (size[0] - eyes_distance) / 2)
        y_offset = kwargs.pop('y_offset', size[1] / 4)
        aux = kwargs.pop('aux', np.array(None))
        resample = kwargs.pop('resample', 0)
            
        if aux.any():
            mid_eye = np.mean([eyes[0], eyes[1]], axis=0)
            aux_distance = kwargs.pop('aux_distance', norm(aux - mid_eye))
            scale_x = norm(eyes[1] - eyes[0]) / eyes_distance
            scale_y = norm(aux - mid_eye) / aux_distance
            scale = np.diag([scale_x, scale_y])
        else:
            scale = norm(eyes[1] - eyes[0]) / eyes_distance

        cos, sin = (eyes[1] - eyes[0]) / norm(eyes[1] - eyes[0])
        sR = np.array([[cos, -sin], [sin, cos]]).dot(scale)
        t = eyes[0] - np.sum(sR * np.array([x_offset, y_offset]), axis=1)
        sRt = np.append(sR, t.reshape(-1, 1), axis=1) # [sR | t]

        coefs = tuple(sRt.ravel())
        matrix = inv(np.append(sRt, np.array([0, 0, 1]).reshape(1, -1), axis=0))

        return self.im.transform(size, PIL.Image.AFFINE, coefs, resample=resample), matrix
    

