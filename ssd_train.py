def train(NUM_CLASSES, nb_epoch, base_lr=3e-4, path_prefix='data/train/', dev="cpu"):
    import keras
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from sklearn.model_selection import train_test_split
    from ssd import SSD300
    from ssd_training import MultiboxLoss
    from ssd_training import Generator
    from ssd_utils import BBoxUtility
    from keras.callbacks import TensorBoard
    plt.rcParams['figure.figsize'] = (8, 8)
    plt.rcParams['image.interpolation'] = 'nearest'
    np.set_printoptions(suppress=True)
    if dev == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # for multi GPUs
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        set_session(tf.Session(config=config))

    # some constants
    NUM_CLASSES = NUM_CLASSES + 1  # 1 means mask
    input_shape = (300, 300, 3)
    # nb_epoch = 5
    # base_lr = 3e-4
    # path_prefix = 'data/train/'  # path to your data

    priors = pickle.load(open('data/prior_boxes_ssd300.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    gt = pickle.load(open('data/train.pkl', 'rb'), encoding='iso-8859-1')  # for python3.x
    lable = pickle.load(open('data/label.pkl', 'rb'), encoding='iso-8859-1')  # for python3.x
    # gt = pickle.load(open('data_convert/train.pkl', 'rb'))
    # keys = sorted(gt.keys())
    # num_train = int(round(0.85 * len(keys)))
    # train_keys = keys[:num_train]
    # val_keys = keys[num_train:]
    # num_val = len(val_keys)
    train_keys, val_keys, train_label, val_label = train_test_split(sorted(lable.keys()), sorted(lable.values()),
                                                                    test_size=0.1, random_state=0)

    num_train = len(train_keys)
    num_val = len(val_keys)
    print(train_keys)
    print(val_keys)

    gen = Generator(gt, bbox_util, 1, path_prefix,
                    train_keys, val_keys,
                    (input_shape[0], input_shape[1]), do_crop=False)

    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('data/weights_SSD300.hdf5', by_name=True)

    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3',]
              # 'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

    # freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
    #           'conv2_1', 'conv2_2', 'pool2',
    #           'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
    #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv4_3_norm',
    #           'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 'fc6', 'fc7',
    #           'conv6_1', 'conv6_2',
    #           'conv7_1', 'conv7_1z', 'conv7_2',
    #           'conv8_1', 'conv8_2',
    #           'pool6'
    #           ]
    for L in model.layers:
        if L.name in freeze:
            L.trainable = False

    def schedule(epoch, decay=0.9):
        return base_lr * decay ** (epoch)

    # checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5
    callbacks = [keras.callbacks.ModelCheckpoint('checkpoints/weights.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True, save_best_only=True, mode='auto', period=1),
                 keras.callbacks.LearningRateScheduler(schedule),TensorBoard(log_dir='logs')]

    optim = keras.optimizers.Adam(lr=base_lr)
    # optim = keras.optimizers.RMSprop(lr=base_lr)
    # optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
    model.compile(optimizer=optim,
                  loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)
    a=0

    history = model.fit_generator(gen.generate(True), steps_per_epoch=num_train,
                                  epochs=nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  validation_steps=num_val,
                                  nb_worker=1)

