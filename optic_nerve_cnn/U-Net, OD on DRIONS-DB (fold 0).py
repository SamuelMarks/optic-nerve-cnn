# coding: utf-8

import glob
import os
from datetime import datetime
# import warnings
# warnings.simplefilter('ignore')
from functools import partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import SGD
from sklearn.model_selection import KFold

from optic_nerve_cnn.utils import log_dice_loss, mean_IOU_gpu, dice_metric, data_generator, th_to_tf_encoding, dice, \
    get_unet_light, get_unet_light_for_fold0, folder
from scripts.dual_IDG import DualImageDataGenerator

K.set_image_dim_ordering('th')


def main(dataset='DRIONS_DB', fold0=True):
    dataset_join = partial(os.path.join, os.path.join(os.path.dirname(os.getcwd()), 'data', 'hdf5_datasets'))
    all_ds = dataset_join('all_data.hdf5')
    h5f = h5py.File(all_ds if os.path.isfile(all_ds) else dataset_join('{dataset}.hdf5'.format(dataset=dataset)), 'r')

    model = get_unet_light_for_fold0(img_rows=256, img_cols=256) if fold0 else get_unet_light()
    model.compile(optimizer=SGD(lr=3e-4, momentum=0.95),
                  loss=log_dice_loss,
                  metrics=[mean_IOU_gpu, dice_metric])

    model.summary()

    # ####
    #
    # Accessing data, preparing train/validation sets division:

    X = h5f['{dataset}/256 px/images'.format(dataset=dataset)]
    Y = h5f['{dataset}/256 px/disc'.format(dataset=dataset)]

    X, Y

    train_idx_cv, test_idx_cv = [], []

    for _train_idx, _test_idx in KFold(len(X), n_folds=5, random_state=1):
        print(_train_idx, _test_idx)
        train_idx_cv.append(_train_idx)
        test_idx_cv.append(_test_idx)

    # train_idx = h5f['RIM-ONE v3/train_idx_driu']
    # test_idx = h5f['RIM-ONE v3/test_idx_driu']

    train_idx = train_idx_cv[0]
    test_idx = test_idx_cv[0]

    len(X), len(train_idx), len(test_idx)

    # #### Generator of augmented data:

    train_idg = DualImageDataGenerator(  # rescale=1/255.0,
        # samplewise_center=True, samplewise_std_normalization=True,
        horizontal_flip=True, vertical_flip=True,
        rotation_range=50, width_shift_range=0.15, height_shift_range=0.15,
        zoom_range=(0.7, 1.3),
        fill_mode='constant', cval=0.0)
    test_idg = DualImageDataGenerator()

    # Testing the data generator and generator for augmented data:

    gen = data_generator(X, Y, 'train', batch_size=1)
    batch = next(gen)
    batch[0].shape

    fig = plt.imshow(np.rollaxis(batch[0][0], 0, 3))
    # plt.colorbar(mappable=fig)
    plt.show()
    plt.imshow(batch[1][0][0], cmap=plt.cm.Greys_r);
    plt.show()

    arch_name = "U-Net light, on {dataset} 256 px fold 0, SGD, high augm, CLAHE, log_dice loss"
    weights_folder = os.path.join(os.path.dirname(os.getcwd()), 'models_weights',
                                  '{},{}'.format(datetime.now().strftime('%d.%m,%H:%M'), arch_name))

    weights_folder

    X_valid, Y_valid = next(data_generator(X, Y, train_or_test='test', batch_size=100, stationary=True))
    plt.imshow(np.rollaxis(X_valid[0], 0, 3))
    plt.show()
    print(X_valid.shape, Y_valid.shape)

    # ### Training
    #
    # If a pretrained model needs to be used, first run "Loading model" section below and then go the "Comprehensive visual check", skipping this section.

    history = model.fit_generator(data_generator(X, Y, train_or_test='train', batch_size=1),
                                  samples_per_epoch=99,
                                  max_q_size=1,

                                  validation_data=(X_valid, Y_valid),
                                  # validation_data=data_generator(X, Y, train_or_test='test', batch_size=1),
                                  # nb_val_samples=100,

                                  nb_epoch=500, verbose=1,

                                  callbacks=[CSVLogger(os.path.join(folder(weights_folder), 'training_log.csv')),
                                             # ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, verbose=1, patience=40),
                                             ModelCheckpoint(os.path.join(folder(weights_folder),
                                                                          # 'weights.ep-{epoch:02d}-val_mean_IOU-{val_mean_IOU_gpu:.2f}_val_loss_{val_loss:.2f}.hdf5',
                                                                          'last_checkpoint.hdf5'),
                                                             monitor='val_loss', mode='min', save_best_only=True,
                                                             save_weights_only=False, verbose=0)])

    # ### Comprehensive visual check

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

    # ### Loading model

    load_model = True  # lock
    if not load_model:
        print('load_model == False')
    else:
        on_model_load(load_model=load_model, model=model)


# ### U-Net architecture
#
# <img src="../pics/u_net_arch.png" width=80%>


def on_model_load(load_model, model):
    # specify file:
    # model_path = '../models_weights/01.11,22:38,U-Net on {dataset} 256 px, Adam, augm, log_dice loss/' \
    #    'weights.ep-20-val_mean_IOU-0.81_val_loss_0.08.hdf5'

    # or get the most recently altered file in a folder:
    model_folder = os.path.join(os.path.dirname(os.getcwd()), 'models_weights',
                                '05.03,02_40,U-Net light, on {dataset} 256 px fold 0, SGD, high augm, CLAHE, log_dice loss')

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
        if 'val_dice' in log.columns:
            print('\n' + 'max val dice: {}, at row:'.format(log['val_dice'].max()))
            print(log.loc[log['val_dice'].argmax()])
