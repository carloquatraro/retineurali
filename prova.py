# Mi modifichi il codice qui sotto in modo che la mia Kfold abbia una accuratezza maggiore?
'''
def cross_valid(model_fun,N_folds,data,masks,dice_coef,h):
    kf = KFold(n_splits=N_folds, shuffle=True)

    results = []
    for f, (train_index, test_index) in enumerate(kf.split(data)):
        dice_c = np.empty(len(test_index))

        testData = data[test_index, :, :, :]
        testMasks = masks[test_index, :, :, :]

        # trainData = data[train_index, :, :, :]
        # trainMasks = masks[train_index, :, :, :]
        trainData, trainMasks =shuffle( data[train_index, :, :, :],masks[train_index, :, :, :])

        model = model_fun(data.shape[1:])
        model.compile(optimizer=Adam(0.001), loss=binary_crossentropy, metrics=['accuracy'])

        history = model.fit(trainData, trainMasks, batch_size=h["batch_size"], epochs=h["epochs"], validation_split=h["validation_split"],verbose=1)

        for n in range(len(test_index)):
            est_mask = np.squeeze(model.predict(testData[n, :, :, :][None, ...]) > 0.7)
            dice_c[n] = dice_coef(tf.convert_to_tensor(testMasks[n, :, :, 0].astype(np.float32)),tf.convert_to_tensor(est_mask.astype(np.float32)))

        results.append(np.mean(dice_c))
        print(np.mean(dice_c))
        del model, trainData, trainMasks, testData, testMasks, est_mask, history
    return results
'''
if acc > acc_best
    acc= acc_best
    hy = hy_best