# Importazione delle librerie da utilizzare
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import tensorflow as tf
from keras import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dense, Flatten, Dropout
from keras.regularizers import l1, l2
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy, categorical_crossentropy
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# Funzione di ridimensionamento delle immagini
def resize_images(X_data, target_size:tuple,iteratore:list, N:int):
    X_dataresized = np.empty((N, target_size[0], target_size[1], 1))
    for img_idx in tqdm(iteratore):
        resized_img = resize(X_data[img_idx,:,:,0], target_size, order=3)
        resized_img = resized_img / np.max(resized_img)
        X_dataresized[img_idx,:,:,:] = resized_img[None,..., None]
    return X_dataresized

# Funzione di definizione del dice coefficient
def dice_coef(y_true, y_pred, smooth=0):
  y_true_f = tf.reshape(tf.cast(y_true, tf.float32),[-1])
  y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
  return (2 * tf.reduce_sum(y_true_f * y_pred_f) + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Funzione di definizione del dice loss
def dice_loss(y_true, y_pred):
  return 1 - dice_coef(y_true, y_pred, smooth=2000)

# Funzione di definizione del modello per la segmentazione
def easy_cnn(input_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, kernel_regularizer=l2(), padding='same'))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same'))

    return model

# Funzione di definizione del modello per la classificazione

def classification_cnn(input_shape):
    model_c = Sequential()
    model_c.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model_c.add(MaxPooling2D(pool_size=(2, 2)))
    model_c.add(Dropout(0.001))
    model_c.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model_c.add(MaxPooling2D(pool_size=(2, 2)))
    model_c.add(Flatten())
    model_c.add(Dropout(0.001))
    model_c.add(Dense(256, activation='relu'))
    model_c.add(Dense(128, activation='relu'))
    model_c.add(Dense(3, activation='sigmoid'))
    model_c.add(Dropout(0.001))
    return model_c


# Funzione di cross validation per la segementazione
def cross_valid(model_fun,N_folds,data,masks,h):

    kf = KFold(n_splits=N_folds, shuffle=True)

    results = []
    for f, (train_index, test_index) in enumerate(kf.split(data)):
        print('f=', f + 1)
        dice_c = np.empty(len(test_index))

        testData = data[test_index, :, :, :]
        testMasks = masks[test_index, :, :, :]

        trainData, trainMasks =shuffle( data[train_index, :, :, :],masks[train_index, :, :, :])

        model = model_fun(data.shape[1:])
        model.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=['accuracy'])

        history = model.fit(trainData, trainMasks, batch_size=h["batch_size"], epochs=h["epochs"], validation_split=h["validation_split"],verbose=1)

        '''
        Questa porzione di codice serve per salvare le informazioni di history in un file csv.
        è stato usato per salvare i dati e riplottarli in una file diverso
        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)

        # or save to csv:
        hist_csv_file = 'history.csv'
        with open(hist_csv_file, mode='w') as file:
            hist_df.to_csv(file)
            file.close()
        '''
        for n in range(len(test_index)):
            est_mask = np.squeeze(model.predict(testData[n, :, :, :][None, ...]) > 0.7)
            dice_c[n] = dice_coef(tf.convert_to_tensor(testMasks[n, :, :, 0].astype(np.float32)),tf.convert_to_tensor(est_mask.astype(np.float32)))

        plt.figure(figsize=(8, 8))
        plt.subplot(121)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylim([0, 2])
        #plt.gca().set_aspect(50)
        plt.subplot(122)
        plt.imshow(testData[n, :, :, :], cmap='gray')
        plt.imshow(est_mask, cmap='gray', alpha=0.4)
        plt.axis('off')
        plt.savefig('Plots/plot_segmentazione_f_' + str(f+1) + '.png', dpi=500)
        plt.show()

        results.append(np.mean(dice_c))
        print(np.mean(dice_c))
        del model, trainData, trainMasks, testData, testMasks

    # Tabella del dice
    plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots()
    ax.axis('off')
    res = pd.DataFrame(results, columns=['Dice coefficient'])
    pd.plotting.table(ax, res, loc='center', cellLoc='center')
    plt.savefig('Plots/dice_coefficients_segmentazione.png', dpi=300)
    plt.show()
    return results


# Funzione di nested cross validation per la classificazione
def Ncross_valid(model_fun, data, masks):

    kf_out = KFold(n_splits=3, shuffle=True)
    kf_in = KFold(n_splits=3, shuffle=True)

    accuracy = []
    precision = []
    recall = []
    best_learning= []
    best_batch = []

    for f, (dev_index, test_index) in enumerate(kf_out.split(data)):
        acc_best = 0
        testData = data[test_index, :, :, :]
        testEtic = masks[test_index]
        devData, devEtic = shuffle(data[dev_index, :, :, :],masks[dev_index])
        for i, (t_index, val_index) in enumerate(kf_in.split(devData)):
            # print('i=', i + 1)
            tData = devData[t_index, :, :, :]
            tEtic = devEtic[t_index]
            valData, valEtic = shuffle(devData[val_index, :, :, :],devEtic[val_index])

            h_params = {'learning_rate': [0.001, 0.01, 0.1]}
            # acc_best = 0
            for g in h_params['learning_rate']:
                # print('g=', g)

                b_size = {'B_size': [32, 64]}
                for k in b_size['B_size']:
                    print ('i=', i+1)
                    print('g=', g)
                    print ('Batch=',k)

                    model = model_fun(tData.shape[1:])
                    model.compile(optimizer=Adam(g), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
                    model.fit(tData, tEtic, batch_size=k, epochs=5, validation_split=0.1,verbose=1)
                    score = model.evaluate(valData, valEtic, verbose=0)

                    if score[1]> acc_best:
                        acc_best = score[1]
                        best_params = {'learning_rate': g, 'batch_size': k}
                        print('Accuracy_best:', acc_best,'Best params: ', best_params)

        model = model_fun(devData.shape[1:])
        model.compile(optimizer=Adam(best_params['learning_rate']), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
        model.fit(devData, devEtic, batch_size=best_params['batch_size'], epochs=5, validation_split=0.1,verbose=1)

        score = model.evaluate(testData, testEtic, verbose=0)

        # print (testEtic.shape)
        # print ( model.predict(testData).shape)

        predizioneEtic = np.argmax(model.predict(testData),axis=-1)

        best_learning.append(best_params['learning_rate'])
        best_batch.append(best_params['batch_size'])
        accuracy.append (accuracy_score(testEtic, predizioneEtic))
        precision.append (precision_score(testEtic, predizioneEtic, average = 'weighted'))
        recall.append (recall_score(testEtic, predizioneEtic, average = 'weighted'))



    # Immagine contenente tabella con metriche migliori di ogni ciclo esterno
    metriche = np.column_stack((accuracy, precision, recall,best_learning,best_batch))
    plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots()
    ax.axis('off')
    res = pd.DataFrame(metriche, columns=['Accuracy', 'Precision', 'Recall','Best learning rate','Best batch size'])
    pd.plotting.table(ax, res, loc='center', cellLoc='center')
    plt.savefig('Plots/metriche_classificazione.png', dpi=500)
    plt.show()

    return accuracy, precision, recall
