from __future__ import division
from PIL import Image
import numpy as np

class FaceImage(object):
    def __init__(self, fp, *args, **kwargs):
        self.im = Image.open(fp)

    def register(self, eyes, size, **kwargs):
        aux = kwargs.pop('aux', None)
        distance = kwargs.pop('distance', size[0] / 2)
        y_offset = kwargs.pop('y_pad', size[1] / 4)
        resample = kwargs.pop('resample', 0)

        if aux == None:
            scale = np.linalg.norm(eyes[1] - eyes[0]) / distance
            x_offset = kwargs.pop('x_offset', (size[0] - distance) / 2)
            p_ref = eyes[0]
        else:
            scale = np.linalg.norm(aux[1] - aux[0]) / distance
            x_offset = kwargs.pop('x_offset', size[0] / 2)
            p_ref = aux[0]

        cos, sin = (eyes[1] - eyes[0]) / np.linalg.norm(eyes[1] - eyes[0])
        sR = scale * np.array([[cos, -sin], [sin, cos]])
        t = p_ref - np.sum(sR * np.array([x_offset, y_offset]), axis=1)
        sRt = np.append(sR, t.reshape(-1, 1), axis=1) # [sr | t]

        coefs = tuple(sRt.ravel())

        return self.im.transform(size, Image.AFFINE, coefs, resample=resample)
