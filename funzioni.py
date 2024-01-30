import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import tensorflow as tf
def resize_images(X_data, target_size:tuple,iteratore:list, N:int):
    X_dataresized = np.empty((N, target_size(0), target_size(1), 1))
    for img_idx in tqdm(iteratore):
        resized_img = resize(X_data[img_idx,:,:,0], target_size, order=3)
        resized_img = resized_img / np.max(resized_img)
        X_dataresized[img_idx,:,:,:] = resized_img[None,..., None]
    return X_dataresized

def dice_coef(y_true, y_pred, smooth=0):
  y_true_f = tf.reshape(tf.cast(y_true, tf.float32),
                        [-1])
  y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
  return (2 * tf.reduce_sum(y_true_f * y_pred_f) + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
  return 1 - dice_coef(y_true, y_pred, smooth=2000)