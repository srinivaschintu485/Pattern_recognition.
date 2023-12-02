
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

import numpy as np
from sklearn.metrics import confusion_matrix
from keras import backend as K
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow as tf

def convert_to_boolean_tensor(img, threshold=0.5):
    return K.cast(K.greater(img, threshold), 'float32')

def confusion_matrix(reference_img, predicted_img):
    return tf.math.confusion_matrix(
        K.flatten(reference_img), K.flatten(predicted_img), num_classes=2
    )

def area_based_metrics1(y_true, y_pred, threshold=0.5):
    predicted_img = convert_to_boolean_tensor(y_true, threshold)
    reference_img = convert_to_boolean_tensor(y_pred, threshold)

    confusion_mat = confusion_matrix(reference_img, predicted_img)

    tp = confusion_mat[1, 1]
    tn = confusion_mat[0, 0]
    fp = confusion_mat[0, 1]
    fn = confusion_mat[1, 0]

    return tp,tn,fp,fn

def accu1(y_true, y_pred):
    tp,tn,fp,fn= area_based_metrics1(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)
def kappa1(y_true, y_pred):
    tp,tn,fp,fn= area_based_metrics1(y_true, y_pred)
    return (tp * tn - fp * fn) / ((tp + fp) * (fn + tn) + (tp + fn) * (fp + tn))
def ce1(y_true, y_pred):
    tp,tn,fp,fn= area_based_metrics1(y_true, y_pred)
    return fp / (tp + fp+1)
def oe1(y_true, y_pred):
    tp,tn,fp,fn= area_based_metrics1(y_true, y_pred)
    return fn / (tp + fn)




