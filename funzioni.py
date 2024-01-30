import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import tensorflow as tf
from keras import Sequential       # for sequential API
from keras.models import Model     # for functional API
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Add, Concatenate
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def resize_images(X_data, target_size:tuple,iteratore:list, N:int):
    X_dataresized = np.empty((N, target_size[0], target_size[1], 1))
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

    model.add(Conv2D(1, kernel_size=(1, 1), activation='softmax', padding='same'))

    return model
def cross_valid(model_fun,N_folds,data,masks,loss,h,names):
    N_folds = 10  # Numero di fold desiderati
    kf = KFold(n_splits=N_folds, shuffle=True)

    results = []
    for f, (train_index, test_index) in enumerate(kf.split(data)):
        scores = np.empty(len(test_index))

        testData = data[test_index, :, :, :]
        testMasks = masks[test_index, :, :, :]
        trainData, trainMasks =shuffle( data[train_index, :, :, :],masks[train_index, :, :, :])

        model = model_fun(data.shape[1:])
        model.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=['accuracy'])

        history = model.fit(trainData, trainMasks, batch_size=h, epochs=h, validation_split=h)

        for n in range(len(test_index)):
            est_mask = np.squeeze(model.predict(testData[n, :, :, :][None, ...]) > 0.7)
            scores[n] = loss(tf.convert_to_tensor(testMasks[n, :, :, 0].astype(np.float32)),tf.convert_to_tensor(est_mask.astype(np.float32)))

        results.append(np.mean(scores))
        print("**********")
        print(np.mean(scores))
        print("**********")
        del model, trainData, trainMasks, testData, testMasks, est_mask, history
   # print(f"Fold {f + 1}, Mean Dice Score: {np.mean(scores)}")

# Restituisci la lista delle performance medie su tutti i fold
    return results

