import tensorflow as tf

from lucid.optvis.param.color import to_valid_rgb
from lucid.optvis.param.spatial import pixel_image, fft_image


def image(w, h=None, batch=None, sd=None, decorrelate=True, fft=True, alpha=False, init_val=None):
    h = h or w
    batch = batch or 1
    channels = 4 if alpha else 3
    shape = [batch, w, h, channels]
    
    if init_val is not None:
        t = tf.Variable( init_val )
        rgb = to_valid_rgb( t[..., :3], decorrelate=True, sigmoid=True )
    else:
        t = fft_image( shape, sd=sd ) if fft else pixel_image( shape, sd=sd )
        rgb = to_valid_rgb( t[..., :3], decorrelate=decorrelate, sigmoid=True )
    
    if alpha:
        a = tf.nn.sigmoid(t[..., 3:])
        return tf.concat([rgb, a], -1)
    
    return rgb
