import os
import cv2
import numpy as np
from sklearn.cluster  import KMeans
import pickle
from tqdm import tqdm, trange

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

# img_paths 一个图像路径的 dict
# labels 41 类图像的 label list
def getClusterCentures(img_paths, labels ,num_words):
    sift_det = cv2.xfeatures2d.SIFT_create()
    des_list = []  # 特征描述
    des_matrix = np.zeros((1, 128))
    for i, label in enumerate(labels):
        print(i)
        for img_path in img_paths[label]:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp, des = sift_det.detectAndCompute(gray, None)
            # print(des)
            if des is not None:
                # print(img_path)
                des_matrix = np.row_stack((des_matrix, des))
            des_list.append(des)

    des_matrix = des_matrix[1:, :]  # the des matrix of sift

    # 计算聚类中心  构造视觉单词词典
    kmeans = KMeans(n_clusters=num_words, random_state=33)
    kmeans.fit(des_matrix)
    centres = kmeans.cluster_centers_  # 视觉聚类中心

    save_obj(centres, 'centres')
    save_obj(des_list, 'des_list')

    return centres, des_list

# 将特征描述转换为特征向量
def des2feature(des, num_words, centures, normalization):
    '''
    des:单幅图像的SIFT特征描述
    num_words:视觉单词数/聚类中心数
    centures:聚类中心坐标   num_words*128
    return: feature vector 1*num_words
    '''
    img_feature_vec = np.zeros((1, num_words), 'float32')
    for i in range(des.shape[0]):
        feature_k_rows = np.ones((num_words, 128), 'float32')
        feature = des[i]
        feature_k_rows = feature_k_rows * feature
        feature_k_rows = np.sum((feature_k_rows - centures) ** 2, 1)
        index = np.argmax(feature_k_rows)
        img_feature_vec[0][index] += 1

    # normalization
    if normalization:
        img_feature_vec = img_feature_vec / np.sqrt(np.sum(img_feature_vec ** 2))
    return img_feature_vec

def get_all_features(des_list, num_words, normalization):
    # 获取所有图片的特征向量
    if normalization:
        allvec = np.zeros((len(des_list), num_words), 'float32')
    else:
        allvec = np.zeros((len(des_list), num_words), 'int32')
    for i in trange(len(des_list)):
        if des_list[i] is not None:
            allvec[i] = des2feature(centures=centres, des=des_list[i], num_words=num_words, normalization=normalization)

    if normalization:
        save_obj(allvec, 'allvec')
    else:
        save_obj(allvec, 'allvec_nonormalization')
    return allvec

def getNearestImg(feature, dataset, num_close):
    '''
    找出目标图像最像的几个
    feature:目标图像特征 [1, num_words]
    dataset:图像数据库  [2050, num_words]
    num_close:最近个数
    return:最相似的几个图像
    '''
    similar = np.sum(feature * dataset, axis=1)
    print(similar.shape)
    dist_index = np.argsort(similar)[::-1]
    return dist_index[:num_close]

def showID(id):
    path = getpath()[os.listdir('./caltech101_subset')[id // 50]][id % 50]
    img = cv2.imread(path)
    cv2.imshow(path, img)
    cv2.waitKey(0)

def tfidf():
    tf = load_obj('allvec_nonormalization')
    df = tf.astype(bool).astype(np.float32)
    df = np.sum(df, axis=0)
    idf = np.log(2050 / (df + np.ones_like(df)))
    tfidf = tf * idf
    print('sqrt', np.sqrt(np.sum(tfidf ** 2, axis=0)).shape)
    tfidf = tfidf / np.expand_dims(np.sqrt(np.sum(tfidf ** 2, axis=1)), axis=-1)
    # print(np.sum(tfidf[1] * tfidf[1]))

    save_obj(tfidf, 'tfidf')

    return tfidf


if __name__ == '__main__':
    centres, des_list = load_obj('centres'), load_obj('des_list')
    tfidf()
    # tf = load_obj('allvec_nonormalization')
    # print(np.argmax(tf))
    # id(des_list, 1000, False)
    tfidf = load_obj('tfidf')
    print(np.sum(tfidf[31] * tfidf[2]), np.sum(tfidf[10] * tfidf[10]))
    print(getNearestImg(tfidf[10], tfidf, 10))
    # showID(855)
    # showID(10)
    # showID(646)


    # print(allvec.shape)



    # print(len(des_list), centres[0])