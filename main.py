import numpy as np
import random as rd
from skimage import io as skio
from sklearn.utils import shuffle
from funzioni import *

# Libraries for Deep Learning models
import tensorflow as tf
from keras import Sequential       # for sequential API
from keras.models import Model     # for functional API
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Add, Concatenate
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from skimage.transform import resize

from sklearn.model_selection import KFold

# Other utilities
from tqdm import tqdm
from matplotlib import pyplot as plt

maindirC="./Lung_dataset/COVID/"
maindirN="./Lung_dataset/Normal/"
maindirVP="./Lung_dataset/Viral_Pneumonia/"

N = 1000
sbj_ids = [*range(1, N+1)]
X_data = np.empty((3*N,299,299,1))
y_data = np.empty((3*N,256,256,1))
input_shapeX = np.array ((299,299,1))
input_shapey = np.array ((256,256,1))

for id in tqdm(sbj_ids):
  #Leggo le immagini covid
  imgC = skio.imread(maindirC+f"/images/COVID-{id}.png").astype('float')
  maskC = skio.imread(maindirC+f"/masks/COVID-{id}.png", as_gray=True).astype('bool')
  maskC = (maskC).astype('float')
  #leggo le immagini normali
  imgN = skio.imread(maindirN + f"/images/Normal-{id}.png").astype('float')
  maskN = skio.imread(maindirN + f"/masks/Normal-{id}.png", as_gray=True).astype('bool')
  maskN = (maskN).astype('float')
  #leggo le immagini vireal pneumonia
  imgVP = skio.imread(maindirVP + f"/images/Viral Pneumonia-{id}.png", as_gray=True).astype('float')
  maskVP = skio.imread(maindirVP + f"/masks/Viral Pneumonia-{id}.png", as_gray=True).astype('bool')
  maskVP = (maskVP).astype('float')
  #Assegno le immagini lette al vettore X_data
  #Prima le imamgini covid
  X_data[id-1,:,:,:] = imgC[None,...,None]
  y_data[id-1,:,:,:] = maskC[None,...,None]/maskC.max()
  #Poi le immagini normali
  X_data[id - 1 + 1000, :, :, :] = imgN[None, ..., None]
  y_data[id - 1 + 1000, :, :, :] = maskN[None, ..., None] / maskN.max()
  #Infine le immagini viral pneumonia
  X_data[id - 1 + 2000, :, :, :] = imgVP[None, ..., None]
  y_data[id - 1 + 2000, :, :, :] = maskVP[None, ..., None] / maskVP.max()

  del imgC, maskC, imgN, maskN, imgVP, maskVP

print(X_data.shape)
print(y_data.shape)

'''
# Provo a printare delle immagini casuali 
numero=rd.randrange(3*N)
print(numero)
plt.imshow(X_data[numero,:,:,0],cmap='gray')
plt.axis('off')
plt.show()
'''

# Ridimensionamento delle immagini e maschere
iteratore = [*range(0, 3*N)]
target_size = (128,128)
X_dataresized = np.empty((3*N,128,128,1))
y_dataresized = np.empty((3*N,128,128,1))

# Creazione X_dataresized
X_dataresize=resize_images(X_data, target_size,iteratore,3*N)
print(X_dataresized.shape)

'''
# Provo a printare delle immagini casuali
numero=rd.randrange(3*N)
print(numero)
plt.imshow(X_dataresized[numero,:,:,0],cmap='gray')
plt.axis('off')
plt.show()
'''

# Creazione y_dataresized
y_dataresized=resize_images(y_data, target_size,iteratore,3*N)
print(y_dataresized.shape)

'''
# Provo a printare delle immagini casuali
numero=rd.randrange(3*N)
print(numero)
plt.imshow(y_dataresized[numero,:,:,0],cmap='gray')
plt.axis('off')
plt.show()
'''


# DICE coefficient
#dice_c = dice_coef(y_true, y_pred, smooth=0)
'''
def dice_coef(y_true, y_pred, smooth=0):
  y_true_f = tf.reshape(tf.cast(y_true, tf.float32),
                        [-1])
  y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
  return (2 * tf.reduce_sum(y_true_f * y_pred_f) + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
'''

'''
'''


model = easy_cnn(X_dataresized.shape[1:])

model.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=['accuracy'])

#Print the summary of the model architecture
model.summary()

N_folds = 10

hyperparams = {
    "batch_size":10,
    "epochs":10,
    "validation_split":0.2,
    "callbacks":[]
}
results = cross_valid(easy_cnn,N_folds,X_dataresized,y_dataresized,binary_crossentropy,hyperparams)


