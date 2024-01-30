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
from keras.losses import SparseCategoricalCrossentropy

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

# Creazione del modello
model = easy_cnn(X_dataresized.shape[1:])

model.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=['accuracy'])

#Print the summary of the model architecture
model.summary()


N_folds = 10
hyperparams = {
    "batch_size":128,
    "epochs":10,
    "validation_split":0.1,
}
results = cross_valid(model,N_folds,X_dataresized,y_dataresized,dice_coef,hyperparams)

nclassi = 3
COVID=np.zeros(N)
Normal=np.ones(N)
ViralPneumonia=2*np.ones(N)
etichette = np.concatenate((COVID,Normal,ViralPneumonia),axis=0)

'''
def cross_valid(model_fun, N_folds, data, masks, loss, names):
  N_folds = 10
  kf = KFold(n_splits=N_folds, shuffle=True)

  results = []
  for f, (dev_index, test_index) in enumerate(kf.split(data)):
    testData = data[test_index, :, :, :]
    testEtic = etichette[test_index, :, :, :]
    devData= data[dev_index,:,:,:]
    devEtic= etichette[dev_index,:,:,:]
    for g, (t_index,val_index) in enumerate(kf.split(devData)):
      tData= devData[t_index,:,:,:]
      tEtic= etichette[t_index,:,:,:]
      valData = devData[val_index, :, :, :]
      valEtic = etichette[val_index, :, :, :]
      
      valData, valEtic = shuffle(devData[val_index, :, :, :], etichette[val_index, :, :, :])


    dice_c = np.empty(len(test_index))

    

    trainData, trainMasks = shuffle(data[dev_index, :, :, :], masks[train_index, :, :, :])

    model = model_fun(data.shape[1:])
    model.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=['accuracy'])

    history = model.fit(trainData, trainMasks, batch_size=h["batch_size"], epochs=h["epochs"],
                        validation_split=h["validation_split"], verbose=1)

    for n in range(len(test_index)):
      est_mask = np.squeeze(model.predict(testData[n, :, :, :][None, ...]) > 0.7)
      dice_c[n] = dice_coef(tf.convert_to_tensor(testMasks[n, :, :, 0].astype(np.float32)),
                            tf.convert_to_tensor(est_mask.astype(np.float32)))

    results.append(np.mean(dice_c))
    print(np.mean(dice_c))
    del model, trainData, trainMasks, testData, testMasks, est_mask, history
  return results
'''
