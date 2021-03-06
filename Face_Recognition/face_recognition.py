import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
# FaceNet to extract face embeddings.
class FaceNet():
    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")
    # Predict embedding from a given face image.根据给定的面部图像预测嵌入。
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = face - (131.0912, 103.8827, 91.4953)
        # Forward pass through deep neural network. The input size should be 224 x 224.通过深度神经网络向前传递。 输入大小应为224 x 224。
        reshaped = np.reshape(face, (1, 3, 224, 224))
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)
    # Get dimensionality of the extracted embeddings.获取提取的嵌入的维数。
    def get_embedding_dimensionality(self):
        return self.dim_embeddings
# The FaceRecognizer model enables supervised face identification.
# FaceRecognizer class provides functionality for predicting the identity of a subject in an image using these embeddings.
# FaceRecognizer类提供使用这些嵌入来预测图像中对象的身份的功能。
class FaceRecognizer():


    # Prepare FaceRecognizer; specify all parameters for face identification.指定用于面部识别的所有参数。
    def __init__(self, num_neighbours=21, max_distance=0.7, min_prob=0.6):
        #super(FaceRecognizer, self).__init__()
        # ToDo: Prepare FaceNet and set all parameters for kNN.准备FaceNet并设置kNN的所有参数。
        # The underlying gallery: class labels and embeddings.类标签和嵌入。
        # The gallery is assembled by collecting and labeling faces in video frames.通过在视频帧中收集和标记面孔组成
        self.labels = []
        self.facenet = FaceNet()
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo:extracts its embedding, and stores it as a training sample in the gallery.
    # ToDo: Update gallery for face identification with a new face with labeled identity.
    def update(self, face, label):

        self.embeddings = np.append(self.embeddings,[self.facenet.predict(face)],axis=0)
        self.labels.append(label)
        # self.save()


    # ToDo: Infer the identity for a new face.
    #  assigns a class label to an aligned face using k-NN.
    def predict(self, face):
        # brute-force search
        # Closed-Set
        embeddings = self.facenet.predict(face)
        dic = np.append(self.embeddings,[embeddings],axis=0)
        nbrs = NearestNeighbors(n_neighbors=self.num_neighbours+1, algorithm='brute').fit(dic)
        distances, indices = nbrs.kneighbors(dic)
        # dindx = np.nonzero(distances[-1])
        dist = distances[-1,1]
        idx = indices[-1,1:]
        label = []
        for i in range(len(idx)):
            index = idx[i]
            label.append(self.labels[index])
        # majority of the k nearest neighbors
        predict_label = Counter(label).most_common(1)[0][0]
        n = Counter(label).most_common(1)[0][1]
        posterior_probability = n/self.num_neighbours
        # Open-Set
        if (dist > self.max_distance or posterior_probability < self.min_prob ):
            predict_label = 'unknow'
        return predict_label,posterior_probability,dist
# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.＃FaceClustering类可根据脸部图像的身份和重新识别来对其进行无监督的聚类。
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.指定聚类算法的所有参数。
    def __init__(self,num_clusters=2, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()
        # The underlying gallery: embeddings without class labels.没有类标签的嵌入。
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))
        # Number of cluster centers for k-means clustering.用于k均值聚类的聚类中心的数量。
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.与不同样本关联的聚类索引。
        self.cluster_membership = np.empty((0, self.facenet.get_embedding_dimensionality()))
        # Maximum number of iterations for k-means clustering.k均值聚类的最大迭代次数。
        self.face = []
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo: extracts and stores an embedding for a new face.
    # ToDo: Update gallery for clustering with a new face.
    def update(self, face):
        # self.embeddings = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings,[self.facenet.predict(face)],axis=0)
        self.face.append(face)

    # ToDo: Perform k-means clustering.//
    def fit(self):
        n = np.shape(self.embeddings)[0]
        dim = self.facenet.get_embedding_dimensionality()
        idx = np.random.randint(n, size=self.num_clusters)
        center_initial = self.embeddings[idx,:]
        dd = np.empty((n, self.num_clusters))
        center = np.empty((self.num_clusters,dim))
        cluster_group = {}
        # cluster_group = np.empty((self.num_clusters,1,dim))
        for i in range(self.num_clusters):
            center[i] = center_initial[i]
            cluster_group[i] = []
        for i in range(self.max_iter):
            self.cluster_membership = []
            for j in range(n):
                for k in range(self.num_clusters):
                    dd[j][k] = distance.euclidean(self.embeddings[j], center[k])
                min = np.argmin(dd[j])
                self.cluster_membership.append(min)
                # cluster_group[min] = np.append(cluster_group[min] ,[self.embeddings[j]],axis=1)
                cluster_group[min].append(self.embeddings[j])
                if (j==n-1):
                    for key in cluster_group.keys():
                        center[key] = np.mean(cluster_group[key], axis=0)
        self.cluster_center = center
        # for i in range(self.max_iter):
        #     self.cluster_membership = []


    # ToDo: Predict nearest cluster center for a given face.
    #  re-identification with a closed-set protocol.
    def predict(self, face):

        embeddings = self.facenet.predict(face)
        #[50,2]
        distance = np.empty((1, self.num_clusters))
        for i in range(self.num_clusters):
            distance[0][i] = np.linalg.norm( embeddings - self.cluster_center[i])

        distances_to_clusters = distance
        predicted_label = np.argmin(distance)
        return predicted_label, distances_to_clusters

