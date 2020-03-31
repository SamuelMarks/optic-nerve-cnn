# coding: utf-8

import glob
import os
from datetime import datetime
# import cv2
from functools import partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import SGD
from sklearn.model_selection import KFold

from optic_nerve_cnn.utils import get_unet_light, log_dice_loss, mean_IOU_gpu, dice_metric, data_generator, folder, \
    th_to_tf_encoding, dice, show_img_pred_corr
from scripts.dual_IDG import DualImageDataGenerator

# import warnings
# warnings.simplefilter('ignore')

K.set_image_dim_ordering('th')


def main(dataset='RIM_ONE_v3', show_best_and_worst=True):
    dataset_join = partial(os.path.join, os.path.join(os.path.dirname(os.getcwd()), 'data', 'hdf5_datasets'))
    all_ds = dataset_join('all_data.hdf5')
    h5f = h5py.File(all_ds if os.path.isfile(all_ds) else dataset_join('{dataset}.hdf5'.format(dataset=dataset)), 'r')

    # ### U-Net architecture
    #
    # <img src="../pics/u_net_arch.png" width=80%>

    model = get_unet_light(img_rows=128, img_cols=128)
    model.compile(optimizer=SGD(lr=3e-4, momentum=0.95),
                  loss=log_dice_loss,
                  metrics=[mean_IOU_gpu, dice_metric])

    model.summary()

    # ####
    #
    # Accessing data, preparing train/validation sets division:

    # Loading full images of desired resolution:
    X = h5f['{dataset}/512 px/images'.format(dataset=dataset)]
    Y = h5f['{dataset}/512 px/cup'.format(dataset=dataset)]
    disc_locations = h5f['{dataset}/512 px/disc_locations'.format(dataset=dataset)]

    X, Y

    test_idx, train_idx = before_train(X=X, h5f=h5f, dataset=dataset)

    # #### Generator of augmented data:

    train_idg = DualImageDataGenerator(  # rescale=1/255.0,
        # samplewise_center=True, samplewise_std_normalization=True,
        horizontal_flip=True, vertical_flip=True,
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=(0.8, 1.2),
        fill_mode='constant', cval=0.0)
    test_idg = DualImageDataGenerator()

    # Comment from: "U-Net, OD cup on DRISHTI-GS, cropped by OD (fold 0)"
    # augm_idg = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True,
    #                              horizontal_flip=True, vertical_flip=True,
    #                              rotation_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
    #                              zoom_range=(1.0, 2.0),
    #                              fill_mode='constant', cval=0.0)
    # if using featurewise_center, featurewise_std_normalization or zca_whitening,
    # augm_idg.fit(X_train, Y_train) is needed first

    # Testing the data generator and generator for augmented data:

    gen = data_generator(X=X, y=Y, resize_to=128, train_or_test='train', batch_size=1,
                         test_idx=test_idx, train_idx=train_idx, disc_locations=disc_locations)
    batch = next(gen)
    batch[0].shape

    fig = plt.imshow(np.rollaxis(batch[0][0], 0, 3))
    # plt.colorbar(mappable=fig)
    plt.show()
    plt.imshow(batch[1][0][0], cmap=plt.cm.Greys_r)
    plt.colorbar()
    plt.show()

    arch_name = "OD Cup, U-Net light on {dataset} 512 px cropped to OD 128 px fold 0, SGD, log_dice loss".format(
        dataset=dataset)
    weights_folder = os.path.join(os.path.dirname(os.getcwd()), 'models_weights',
                                  '{},{}'.format(datetime.now().strftime('%d.%m,%H:%M'), arch_name))

    weights_folder

    X_valid, Y_valid = next(
        data_generator(X=X, y=Y, test_idx=test_idx, train_idx=train_idx, test_idg=test_idg, train_idg=train_idg,
                       train_or_test='test', batch_size=100, stationary=True, disc_locations=disc_locations)
    )
    plt.imshow(np.rollaxis(X_valid[0], 0, 3))
    plt.show()
    print(X_valid.shape, Y_valid.shape)

    # ### Training
    #
    # If a pretrained model needs to be used, first run "Loading model" section below and then go the "Comprehensive visual check", skipping this section.

    history = model.fit_generator(
        data_generator(X=X, y=Y, test_idx=test_idx, train_idx=train_idx, test_idg=test_idg, train_idg=train_idg,
                       train_or_test='train', batch_size=1, disc_locations=disc_locations),
        samples_per_epoch=len(train_idx),
        max_q_size=1,

        validation_data=(X_valid, Y_valid),
        # validation_data=data_generator(X, Y, train_or_test='test', batch_size=1),
        # nb_val_samples=100,

        nb_epoch=500, verbose=1,

        callbacks=[
            CSVLogger(os.path.join(folder(weights_folder), 'training_log.csv'), append=True),
            # ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, verbose=1, patience=50),
            ModelCheckpoint(os.path.join(folder(weights_folder),
                                         # 'weights.ep-{epoch:02d}-val_mean_IOU-{val_mean_IOU_gpu:.2f}_val_loss_{val_loss:.2f}.hdf5',
                                         'last_checkpoint.hdf5'),
                            monitor='val_loss', mode='min', save_best_only=True,
                            save_weights_only=False, verbose=0)])

    pred_iou, pred_dice = [], []

    if K.backend() == 'tensorflow':
        sess = K.get_session()

    for i, img_no in enumerate(test_idx):
        print('image #{}'.format(img_no))
        img = X[img_no]
        batch_X = X_valid[i:i + 1]
        batch_y = Y_valid[i:i + 1]

        pred = (model.predict(batch_X)[0, 0] > 0.5).astype(np.float64)
        # corr = Y[img_no][..., 0]
        corr = th_to_tf_encoding(batch_y)[0, ..., 0]

        # mean filtering:
        # pred = mh.mean_filter(pred, Bc=mh.disk(10)) > 0.5

        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(pred, cmap=plt.cm.Greys_r)
        ax.set_title('Predicted')
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(corr, cmap=plt.cm.Greys_r)
        ax.set_title('Correct')
        ax = fig.add_subplot(1, 3, 3)
        # ax.imshow(img)
        ax.imshow(th_to_tf_encoding(batch_X)[0])
        ax.set_title('Image')
        plt.show()

        if K.backend() == 'tensorflow':
            cur_iou = mean_IOU_gpu(pred[None, None, ...], corr[None, None, ...]).eval(session=sess)
            cur_dice = dice(pred[None, None, ...], corr[None, None, ...]).eval(session=sess)
        else:
            cur_iou = mean_IOU_gpu(pred[None, None, ...], corr[None, None, ...]).eval()
            cur_dice = dice(pred[None, None, ...], corr[None, None, ...]).eval()
        print('IOU: {}\nDice: {}'.format(cur_iou, cur_dice))
        pred_iou.append(cur_iou)
        pred_dice.append(cur_dice)

    # Acquiring scores for the validation set:

    print(np.mean(pred_iou))
    print(np.mean(pred_dice))

    # Showing the best and the worst cases:
    if show_best_and_worst:
        best_idx = np.argmax(pred_iou)
        worst_idx = np.argmin(pred_iou)
        show_img_pred_corr(best_idx, 'best', test_idx=test_idx, disc_locations=disc_locations, X=X, Y=Y, model=model,
                           test_idg=test_idg, dataset=dataset)
        print('IOU: {}, Dice: {} (best)'.format(pred_iou[best_idx], pred_dice[best_idx]))
        show_img_pred_corr(worst_idx, 'worst', test_idx=test_idx, disc_locations=disc_locations, X=X, Y=Y, model=model,
                           test_idg=test_idg, dataset=dataset)
        print('IOU: {}, Dice: {} (worst)'.format(pred_iou[worst_idx], pred_dice[worst_idx]))

    # ### Loading model

    load_model = True  # lock
    if not load_model:
        print('load_model == False')
    else:
        on_model_load(load_model=load_model, model=model, dataset=dataset)


def before_train(X, h5f, dataset):
    n_folds = 5
    train_idx_cv, test_idx_cv = [], []

    for _train_idx, _test_idx in KFold(X.shape[0], n_folds):
        train_idx_cv.append(_train_idx)
        test_idx_cv.append(_test_idx)

    # train_idx = h5f['{dataset}/train_idx_driu'.format(dataset=dataset)]
    # test_idx = h5f['{dataset}/test_idx_driu'.format(dataset=dataset)]

    train_idx = train_idx_cv[0]
    test_idx = test_idx_cv[0]

    len(X), len(train_idx), len(test_idx)

    return test_idx, train_idx


def on_model_load(load_model, model, dataset):
    # specify file:
    # model_path = '../models_weights/01.11,22:38,U-Net on DRIONS-DB 256 px, Adam, augm, log_dice loss/' \
    #    'weights.ep-20-val_mean_IOU-0.81_val_loss_0.08.hdf5'
    # or get the most recent file in a folder:
    model_folder = weights_folder = os.path.join(os.path.dirname(os.getcwd()), 'models_weights',
                                                 '01.03,10_33,OD Cup, U-Net light on {dataset} 512 px cropped to OD 128 px fold 0, SGD, log_dice loss'.format(
                                                     dataset=dataset))

    # model_folder = weights_folder

    model_path = max(glob.glob(os.path.join(model_folder, '*.hdf5')), key=os.path.getctime)
    if load_model and not os.path.exists(model_path):
        raise Exception('`model_path` does not exist')
    print('Loading weights from', model_path)

    if load_model:
        # with open(model_path + ' arch.json') as arch_file:
        #    json_string = arch_file.read()
        # new_model = model_from_json(json_string)
        model.load_weights(model_path)

    # Reading log statistics
    import pandas as pd

    log_path = os.path.join(model_folder, 'training_log.csv')
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        if log['epoch'].dtype != 'int64':
            log = log.loc[log.epoch != 'epoch']
        print('\nmax val mean IOU: {}, at row:'.format(log['val_mean_IOU_gpu'].max()))
        print(log.loc[log['val_mean_IOU_gpu'].argmax()])
        if 'val_dice_metric' in log.columns:
            print('\n' + 'max val dice_metric: {}, at row:'.format(log['val_dice_metric'].max()))
            print(log.loc[log['val_dice_metric'].argmax()])
