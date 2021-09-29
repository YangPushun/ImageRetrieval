import  os
import  cv2
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
from    autoencoder import AE
import  datetime

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

model = AE(h_dim=1000)

def norm_img(image):
    image = image/127.5 - 1.
    return image

def denorm_img(norm):
    return tf.cast(tf.clip_by_value((norm + 1)*127.5, 0, 255), tf.int32)

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
    imgs = tf.zeros([1, 128, 128, 3], dtype=tf.float32)
    for i, label in enumerate(labels):
        for img_path in img_paths[label]:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = norm_img(tf.reshape(tf.convert_to_tensor(img, dtype=tf.float32), [1, 128, 128, 3]))
            imgs = tf.concat([imgs, img], axis=0)



    return imgs[1:,:,:]

def train(train_db, batchsz):
    lr = 1e-3

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'D:/PCProjects/Pose_Guide_data/logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    train_db = tf.data.Dataset.from_tensor_slices(train_db)
    train_db = train_db.repeat(10000).shuffle(2050).batch(batchsz)
    model.build(input_shape=(None, 128, 128, 3))

    optimizer = tf.optimizers.Adam(lr=lr)

    for step, x in enumerate(train_db):

        x = tf.reshape(x, [-1, 128, 128, 3])

        with tf.GradientTape() as tape:
            x_rec_logits = model(x)

            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(step, float(rec_loss))
            with summary_writer.as_default():
                # print('x:', tf.reduce_min(self.x), tf.reduce_max(self.x))
                # print('G1:', tf.reduce_min(G1), tf.reduce_max(G1))
                # tf.summary.scalar('train-g_loss1', float(g_loss1), step=step)
                tf.summary.image("x_rec_logits:", tf.cast(denorm_img(tf.clip_by_value(x_rec_logits[0:5, :, :, :], -1.0, 1.0)), dtype=tf.uint8).numpy(),
                                 step=step)
                tf.summary.image("x:",
                                 tf.cast(denorm_img(tf.clip_by_value(x[0:5, :, :, :], -1.0, 1.0)), dtype=tf.uint8).numpy(),
                                 step=step)

    model.save_weights('./model/weights.ckpt')



if __name__ == '__main__':
    imgs = build_dataset()
    train(imgs, 16)