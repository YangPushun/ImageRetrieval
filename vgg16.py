import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, models, Input, optimizers
from    tensorflow.keras.models import Model, load_model
from    tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import  os
import  cv2
import  numpy as np
import  pickle

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)

def VGG16():
    base_model = keras.applications.VGG16(weights='imagenet')
    # base_model.summary()

    layers_list = [l for l in base_model.layers]
    x = layers_list[0](layers_list[0].output)
    for i in range(1, len(layers_list)):
        if layers_list[i].name == 'fc2':
            x = Dense(128, activation='relu', name='fc2')(x)
            continue
        elif layers_list[i].name == 'predictions':
            x = Dense(41, activation=None, name='predictions')(x)
            continue
        x = layers_list[i](x)

    new_model = Model(layers_list[0].input, x)
    return new_model


# def VGG16(nb_classes, input_shape):
#     input_tensor = Input(shape=input_shape)
#     # 1st block
#     x = Conv2D(64, (3,3), activation='relu', padding='same',name='conv1a')(input_tensor)
#     x = Conv2D(64, (3,3), activation='relu', padding='same',name='conv1b')(x)
#     x = MaxPooling2D((2,2), strides=(2,2), name = 'pool1')(x)
#     # 2nd block
#     x = Conv2D(128, (3,3), activation='relu', padding='same',name='conv2a')(x)
#     x = Conv2D(128, (3,3), activation='relu', padding='same',name='conv2b')(x)
#     x = MaxPooling2D((2,2), strides=(2,2), name = 'pool2')(x)
#     # 3rd block
#     x = Conv2D(256, (3,3), activation='relu', padding='same',name='conv3a')(x)
#     x = Conv2D(256, (3,3), activation='relu', padding='same',name='conv3b')(x)
#     x = Conv2D(256, (3,3), activation='relu', padding='same',name='conv3c')(x)
#     x = MaxPooling2D((2,2), strides=(2,2), name = 'pool3')(x)
#     # 4th block
#     x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv4a')(x)
#     x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv4b')(x)
#     x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv4c')(x)
#     x = MaxPooling2D((2,2), strides=(2,2), name = 'pool4')(x)
#     # 5th block
#     x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv5a')(x)
#     x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv5b')(x)
#     x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv5c')(x)
#     x = MaxPooling2D((2,2), strides=(2,2), name = 'pool5')(x)
#     # full connection
#     x = Flatten()(x)
#     x = Dense(4096, activation='relu',  name='fc6')(x)
#     # x = Dropout(0.5)(x)
#     x = Dense(128, activation='relu', name='fc7')(x)
#     # x = Dropout(0.5)(x)
#     output_tensor = Dense(nb_classes, activation='softmax', name='fc8')(x)
#
#     model = Model(input_tensor, output_tensor)
#     return model
#
# model=VGG16(41, (224, 224, 3))
# model.summary()

def norm_img(image):
    image = image/127.5 - 1.
    return image

def denorm_img(norm):
    return tf.cast(tf.clip_by_value((norm + 1)*127.5, 0, 255), tf.int32)

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def getpath():
    path1 = './caltech101_subset'
    subdirs = os.listdir(path1)
    pics = {}
    for subset in subdirs:
        pics[subset] = [os.path.join(path1, subset, dir) for dir in os.listdir(os.path.join(path1, subset))]
    # print(pics['airplanes'])

    return pics

def build_dataset():
    img_paths = getpath()
    labels = os.listdir('./caltech101_subset')
    imgs = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
    for i, label in enumerate(labels):
        for img_path in img_paths[label]:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = norm_img(tf.reshape(tf.convert_to_tensor(img, dtype=tf.float32), [1, 224, 224, 3]))
            imgs = tf.concat([imgs, img], axis=0)

    labels = tf.zeros([50], dtype=tf.int32)
    for i in range(1, 41):
        labels = tf.concat([labels, tf.ones([50], dtype=tf.int32) * i], axis=0)

    # labels = tf.one_hot(labels, depth=41)
    train_db = tf.data.Dataset.from_tensor_slices((imgs[1:,:,:], labels))

    return train_db

def train(train_db, test_db, newvgg):
    # train_db = train_db.shuffle(1000).batch(16)
    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(15):

        for step, (x, y) in enumerate(train_db):

            # print(x.shape)

            with tf.GradientTape() as tape:
                out = newvgg(x)
                # print(out.shape, y.shape)
                y_onehot = tf.one_hot(y, depth=41)

                loss = tf.losses.categorical_crossentropy(y, out, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, newvgg.trainable_variables)
            optimizer.apply_gradients(zip(grads, newvgg.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
        if (epoch + 1) % 5 == 0:
            newvgg.save("./model/newvgg_" + str(epoch) + '.h5')
        test(test_db, newvgg)

def test(test_db, newvgg):
    total_num = 0
    total_correct = 0
    for x, y in test_db:
        logits = newvgg(x)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        # print(pred.shape, y.shape)

        correct = tf.cast(tf.equal(pred, tf.cast(y, dtype=tf.int32)), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    print('acc:', acc)

def get_all_features(train_db, vgg):
    train_db = train_db.repeat(1).batch(10)
    layers_list = [l for l in vgg.layers]
    features = tf.zeros([1, 128], dtype=tf.float32)
    for x, y in train_db:
        out = layers_list[0](x)
        for i in range(1, len(layers_list) - 1):
           out = layers_list[i](out)
        features = tf.concat([features, out], axis=0)

    features = features[1:, :].numpy()
    save_obj(features, 'vgg_features')

    return features

def getNearestImg(feature, dataset, num_close):
    features = tf.ones((dataset.shape[0], len(feature)), 'float32')
    features = features * feature
    dist = tf.reduce_sum((features - dataset) ** 2, axis=1)
    dist_index = np.argsort(dist)
    return dist_index[:num_close]

if __name__ == '__main__':
    print('start')
    # train_db = build_dataset()
    # train_db = train_db.shuffle(1000).batch(16)
    # test_db = build_dataset()
    # test_db = test_db.shuffle(2050).batch(32)
    # newvgg = VGG16()
    # newvgg.save('./model/newvgg/newvgg.h5')

    newvgg = load_model('./model/newvgg_14.h5')
    newvgg.summary()
    # train(train_db, test_db, newvgg)
    # test(train_db, newvgg)
    # get_all_features(train_db, newvgg)
    # features = load_obj('vgg_features')
    # print(getNearestImg(features[2], features, 10))