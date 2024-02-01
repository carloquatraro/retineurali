# Importazione delle librerie da utilizzare
import numpy as np
import random as rd
from skimage import io as skio
from sklearn.utils import shuffle
from funzioni import *
import tensorflow as tf
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Add, Concatenate
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from skimage.transform import resize
from keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import KFold
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# Directories del dataset
maindirC="./Lung_dataset/COVID/"
maindirN="./Lung_dataset/Normal/"
maindirVP="./Lung_dataset/Viral_Pneumonia/"

# Creazione degli array per le immagini e le maschere
N = 1000
sbj_ids = [*range(1, N+1)]
X_data = np.empty((3*N,299,299,1))
y_data = np.empty((3*N,256,256,1))
input_shapeX = np.array ((299,299,1))
input_shapey = np.array ((256,256,1))

# Caricamento delle immagini e delle maschere
for id in tqdm(sbj_ids):

  # Leggo le immagini covid
  imgC = skio.imread(maindirC+f"/images/COVID-{id}.png").astype('float')
  maskC = skio.imread(maindirC+f"/masks/COVID-{id}.png", as_gray=True).astype('bool')
  maskC = (maskC).astype('float')

  # Leggo le immagini normali
  imgN = skio.imread(maindirN + f"/images/Normal-{id}.png").astype('float')
  maskN = skio.imread(maindirN + f"/masks/Normal-{id}.png", as_gray=True).astype('bool')
  maskN = (maskN).astype('float')

  # Leggo le immagini viral pneumonia
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

# Verifico le dimensioni degli array
print(X_data.shape)
print(y_data.shape)

'''
# Print casuale di una immagine 
numero=rd.randrange(3*N)
print(numero)
plt.imshow(X_data[numero,:,:,0],cmap='gray')
plt.axis('off')
plt.show()
'''

# Ridimensionamento delle immagini e maschere
M = 3000
iteratore = [*range(0, M)]
target_size = (128,128)
X_dataresized = np.empty((M,128,128,1))
y_dataresized = np.empty((M,128,128,1))

# Creazione e riempimento di X_dataresized con le immagini ridimensionate
X_dataresized=resize_images(X_data, target_size,iteratore,M)
print(X_dataresized.shape)

'''
# Print casuale di una immagine ridimensionata 
numero=rd.randrange(M)
print(numero)
plt.imshow(X_dataresized[numero,:,:,0],cmap='gray')
plt.axis('off')
plt.show()
'''

# Creazione e riempimento di y_dataresized con le maschere ridimensionate
y_dataresized=resize_images(y_data, target_size,iteratore,M)
print(y_dataresized.shape)

'''
# Print casuale di una maschera ridimensionata
numero=rd.randrange(M)
print(numero)
plt.imshow(y_dataresized[numero,:,:,0],cmap='gray')
plt.axis('off')
plt.show()
'''

# Creazione del modello per segmentazione e print del suo summary
model = easy_cnn(X_dataresized.shape[1:])
model.summary()

# Definizione degli iperparametri utilizzati per la segmentazione
N_folds = 5
hyperparams = {
    "batch_size":10,
    "epochs":10,
    "validation_split":0.1,
}

# Segmentazione
results = cross_valid(easy_cnn,N_folds,X_dataresized,y_dataresized,hyperparams)

# Creazione del modello per classificazione e print del suo summary
model_c = classification_cnn(X_dataresized.shape[1:])
model_c.summary()

# Definizione del numero di classi e creazione del vettore contente le etichette
nclassi = 3
COVID=np.zeros(N)
Normal=np.ones(N)
ViralPneumonia=2*np.ones(N)
etichette = np.concatenate((COVID,Normal,ViralPneumonia),axis=0)

# Classificazione con estrapolazione delle metriche
accuracy,precision,recall = Ncross_valid(classification_cnn, X_dataresized, etichette)

# Media delle metriche richieste e print
average_accuracy = np.mean(accuracy)
average_precision = np.mean(precision)
average_recall = np.mean(recall)

print(f'Accuracy: {average_accuracy}')
print(f'Precision: {average_precision}')
print(f'Recall: {average_recall}')




