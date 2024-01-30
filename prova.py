import numpy as np
import random as rd
from skimage import io as skio
from sklearn.utils import shuffle
import albumentations as A

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

# Ridimensionamento delle immagini
iteratore = [*range(0, 3*N)]
target_size = (128,128)
X_dataresized = np.empty((3*N,128,128,1))
y_dataresized = np.empty((3*N,128,128,1))

def resize_images(X_data, target_size):
  for img_idx in tqdm(iteratore):
    resized_img = resize(X_data[img_idx,:,:,0], target_size, order=3)
    resized_img = resized_img / np.max(resized_img)
    X_dataresized[img_idx,:,:,:] = resized_img[None,..., None]

  return X_dataresized

print(X_dataresized.shape)

resize_images(y_data, target_size)
print(y_dataresized.shape)

def dice_coef(y_true, y_pred, smooth=0):
  y_true_f = tf.reshape(tf.cast(y_true, tf.float32),
                        [-1])
  y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
  return (2 * tf.reduce_sum(y_true_f * y_pred_f) + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
def dice_loss(y_true, y_pred):
  return 1 - dice_coef(y_true, y_pred, smooth=2000)

def easy_cnn(input_shape):
    model = Sequential()

    # Encoder (facciamo una compressione)
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, kernel_regularizer=l2(), padding='same'))
    # convolvo la mia immagine 32 volte con un kernel 3x3, poi passiamo l'input shape che è il primo layer e poi c'è la regolarizzazione che
    # è un modo per evitare che i parametri della rete crescano troppo e serve per evitare a generalizzare, con padding = same chiediamo in
    # uscita una feature map che abbia le stesse dimensioni.
    model.add(BatchNormalization()) # BatchNormalization è un layer che serve a normalizzare i valori di tutto il batch. Il batch
    # è un insieme di dati che passo alla mia rete tutti insieme e su cui lei andrà a fare predizione prima di aggiornare i pesi. In questo caso
    # la batch è un insieme di immagini, ogni immagine potrebbe essere normalizzata ma l'insieme delle immagini no. Allora il layer BatchNormalization
    # normalizza tutto il batch. Questo serve a generalizzare.
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling è un sottocampionamento, da un intorno 2x2 estrae un campione che sarà il valore massimo.
    # Questo perchè sto cercando di comprimere la mia immagine per estrarre delle feature che siano relative alla regione.

    # C'è un'altra compressione a 64, tipicamente più vado in profondità più le informazioni sono specifiche. Avrei potuto fare un ulteriore compressione
    # a 128.
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Decoder (qui facciamo la seconda parte del modello che fa l'upsampling). E' speculare, ribaltimao il primo modello. Di fatto è quasi un copia incolla
    # ma al posto di fare un sottocampionamento (maxpooling) si fa un sovracampionamento. Da verificare che le dimensioni di ingresso e uscita siano
    # uguali. Nella fase di convuluzione lo dico con padding=same, nella fase di upsampling o downsampling devo stare attento io.
    # NEL NOSTRO CASO: le immagini sono 299x299, quando facciamo un maxpooling con kernel 2x2 otteniamo 149x149 mentre con l'opsampling 2x2 verranno
    # delle immagini che sono 298x298 quindi, siccome abbiamo immagini con dimensioni dispari, se facciamo down e upsampling con kernel pari otteniamo
    # una maschera che non ha le stesse dimensioni dell'immagine, si risplverebbe utilizzando un kernel dispari.
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    # Output layer
    model.add(Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same'))
    # all'ultimo mettiamo un output con una sigmoide perchè è una funzione che permette di avere delle probabilità in uscita

    return model

model = easy_cnn(X_data.shape[1:])

#Compile the model
model.compile(optimizer=Adam(0.001), loss=dice_loss, metrics=['accuracy'])

#Print the summary of the model architecture
model.summary()

def cross_valid(model_fun,N_folds,data,masks,loss,h,names):

    kf = KFold(n_splits=N_folds,shuffle=False) #KFold restituisce ogni volta una coppia di indici che dobbiamo utilizzare per test e training di volta
    # in volta, è quindi una funzione che, applicata sui nostri dati, restituisce tante coppie. Abbiamo train index che è una lista
    # con gli indici dei soggetti di training
    # e test index che è una lista con gli indici dei soggetti di test (al massimo potrebbe essere uno valore). Possiamo usare queste liste di indici
    # per andare a prendere delle porzioni specifiche del nostro dataset
    results = []

    # Questo codice eseguirà la validazione incrociata su K fold e raccoglierà i risultati in una lista chiamata results


    for f, (train_index, test_index) in enumerate(kf.split(data)):
#      if f>0: break
      print("Fold #"+str(f+1))

      scores = np.empty(len(test_index)) # è una funzione di NumPy che crea un nuovo array con dimensioni specificate

#La variabile f rappresenta l'indice del fold corrente, mentre train_index e test_index sono gli indici per i set di addestramento e test relativi al fold corrente. La variabile
# scores sembra essere utilizzata per memorizzare i risultati (ad esempio, le metriche di valutazione) ottenuti per ogni istanza del set di test

     # Splitting data into test and train
      testData = X_dataresized[test_index,:,:,:]
      testMasks = y_dataresized[test_index,:,:,:]   #Sembrerebbe che tu stia estraendo i dati di test (testData) e le relative maschere (testMasks) basandoti sugli indici ottenuti dalla
                                            # K-fold cross-validation. Questo passaggio è comune quando si esegue la validazione incrociata per separare il dataset in un set di
                                            # addestramento e uno di test per ogni fold.


      # Shuffling training data
      trainData, trainMasks = shuffle(X_dataresized[train_index,:,:,:],y_dataresized[train_index,:,:,:])

      # Single-fold training
      model = model_fun(X_dataresized.shape[1:]) # definisamo il modello
      model.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=['accuracy']) # compiliamo
      history = model.fit(trainData,trainMasks,batch_size=h["batch_size"],epochs=h["epochs"],validation_split=h["validation_split"],callbacks=h["callbacks"])
      # facciamo il fit, passando le maschere e tutti gli iperparametri che servono. Validation_split, e callbacks provenienti dal dizionario h, se validation_split è impostato su 0.1,
      # il 10% dei dati sarà utilizzato per la validazione; I callback sono oggetti che possono eseguire azioni specifiche in determinati punti durante l'addestramento, ad esempio
      # salvare i pesi del modello o interrompere l'addestramento prematuramente.
      # dopo l'addestramento, history conterrà informazioni sull'andamento dell'addestramento, come le perdite e le metriche, che puoi utilizzare per l'analisi e la visualizzazione
      # dei risultati

      # se mai, dopo che ho fatto il fit su i miei dati di training e ho utilizzato validation split quindi al mio interno li ho ulteriormente suddivisi
      # devo fare manualmente il test sui pazienti che mi sono rimasti fuori quindi all'interno di testdata (precedentemente estratta) faccio la predizione
      # su ciascuno di loro (soggetti di test) e applico la mia funzione di loss (qui Paolo ha preso un array ed ha archiviato tutte le loss su ognuno dei
      #soggetti di test per poi fare la media)
      # Single-fold testing
      for n in range(len(test_index)):
          est_mask = np.squeeze(model.predict(testData[n,:,:,:][None,...])>0.7)  # Effettua la predizione del modello sulla singola immagine di test (testData[n,:,:,:]) e applica una
                                                                                 # soglia (0.7 nel tuo caso) per ottenere una maschera binaria (True o False). La funzione np.squeez
                                                                                 # e viene utilizzata per rimuovere le dimensioni superflue.
          #est_mask = region_refining(est_mask)
          scores[n] = loss(tf.convert_to_tensor(testMasks[n,:,:,0].astype(np.float32)),tf.convert_to_tensor(est_mask.astype(np.float32)))
          #La riga di codice che hai fornito sta calcolando il punteggio utilizzando la tua funzione di perdita (loss). La funzione di perdita sta confrontando la maschera di ground
          #truth (testMasks[n,:,:,0]) con la maschera stimata (est_mask). convert_to_tensor converte la maschera in un tensore di tensorflow.


      results.append(np.mean(scores))
      print("**********")
      print(np.mean(scores))
      print("**********")
      del model, trainData, trainMasks, testData, testMasks, est_mask, history

    return results