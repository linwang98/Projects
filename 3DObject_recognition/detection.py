import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import cv2
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter



class matlab:

    def __init__(self, index):
        mat_path = "D:/PycharmProjects/pythonProject/exercise 1/files/example" + str(index) + "kinect.mat"
        self.mat = sio.loadmat(mat_path)
        self.index = index
        self.cloud = self.mat['cloud' + str(self.index)]
        self.amplitudes = self.mat['amplitudes' + str(self.index)]
        self.distances = self.mat['distances' + str(self.index)]

    def get_valid_cloud(self):
        cloud = self.cloud
        cloud = cloud.reshape((-1, 3))
        non_zero = np.nonzero(cloud[:, 2])
        cloud = cloud[non_zero, :].squeeze()
        print(cloud.shape)
        return cloud

    def plane_model(self, points):
        p1, p2, p3 = points[0, :], points[1, :], points[2, :]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        d = np.dot(n, p1)
        return n, d

    def ransac(self, threshold, iterations=100):
        valid_cloud = self.get_valid_cloud()
        max_inliers = 0
        best_model = None
        num_points = valid_cloud.shape[0]
        for _ in range(iterations):
            index = np.random.choice(num_points, size=3, replace=False)
            threepoints = valid_cloud[index, :]
            n, d = self.plane_model(threepoints)
            distances = np.abs(np.inner(valid_cloud, n) - d) / np.linalg.norm(n)
            n_inliers = np.count_nonzero(distances < threshold)
            if n_inliers > max_inliers:
                best_model = (n, d)
                max_inliers = n_inliers
        print(max_inliers)
        return best_model

    def get_floor_plane(self, ran_threshold, top_plane):
        n, d = self.ransac(threshold=ran_threshold)
        full_cloud = self.cloud
        distances = np.abs(np.inner(full_cloud, n) - d) / np.linalg.norm(n)
        #rest = np.where(distances < ran_threshold, 1, 0)
        floor_plane = np.where(distances < ran_threshold, 1,0)
        plt.subplot(131)
        plt.imshow(floor_plane)
        plt.title('floor_plane', fontsize=10)
        plt.subplot(132)
        #Filtering on the Mask Image morphological operators# 删除小黑洞
        erosion=scipy.ndimage.morphology.binary_closing(floor_plane)
        # 删除小白色区域
        erosion = ndimage.binary_opening(erosion)
        plt.imshow(erosion)
        plt.title('erosion', fontsize=10)
        plt.subplot(133)
        plt.imshow(floor_plane-top_plane)
        plt.title('height', fontsize=10)
        plt.subplot(133)
        plt.show()
        return floor_plane,n,d

    def get_top_plane(self, ran_threshold):
        n, d = self.ransac(threshold=ran_threshold)
        full_cloud = self.cloud
        distances = np.abs(np.inner(full_cloud, n) - d) / np.linalg.norm(n)
        #rest = np.where(distances<ran_threshold,1,0)
        top_plane = np.where(distances<ran_threshold,0,1)
        plt.subplot(221)
        plt.imshow(top_plane)
        plt.title('top_plane', fontsize=10)
        plt.subplot(222)
        #Filtering on the Mask Image morphological operators
        erosion=scipy.ndimage.morphology.binary_closing(top_plane)
        erosion = ndimage.binary_opening(erosion)
        plt.imshow(erosion)
        plt.title('erosion', fontsize=10)
        plt.subplot(223)#nd_labels
        #  erosion=gaussian_filter(erosion, sigma=3) gaussian_filter
        label_im, nd_labels = ndimage.label(erosion)
        #find the sizes of each labeled region
        sizes = ndimage.sum(erosion, label_im, range(nd_labels + 1))
        max_label = np.where(sizes == sizes.max())
        #Clean up small connect components:
        mask_size = sizes < 1000
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
        plt.imshow(label_im) 
        plt.title('Clean up', fontsize=10)
        plt.subplot(224)
        #Now reassign labels with np.searchsorted:
        labels = np.unique(label_im)
        binary_img= np.searchsorted(labels, label_im)
        #Select the biggest connected component
        binary_img[binary_img < binary_img.max()]=0
        binary_img[binary_img >= binary_img.max()]=1
        plt.imshow(binary_img)
        plt.title('filtered', fontsize=10)
        plt.show()
        return binary_img

    def visualization(self, data_cloud):
        im = ndimage.rotate(data_cloud, 15, mode='constant')
        im = ndimage.gaussian_filter(im, 8)
        sx = ndimage.sobel(im, axis=0, mode='constant')
        sy = ndimage.sobel(im, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        plt.figure(figsize=(16, 5))
        plt.subplot(141)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('square', fontsize=20)
        plt.subplot(142)
        plt.imshow(sx)
        plt.axis('off')
        plt.title('Sobel (x direction)', fontsize=20)
        plt.subplot(143)
        plt.imshow(sob)
        plt.axis('off')
        plt.title('Sobel filter', fontsize=20)
        plt.show()

    def corner(self, top_mask):
        max = 0
        min = 1000
        m = 1000000
        n = 1000
        for i in range(414):
            for j in range(512):
                if (top_mask[i][j] == 1):

                    distance = i**2+j**2

                    if (min > j):
                        min = j
                        a = j
                        b = i

                    if (m > distance):
                        m = distance
                        c = j
                        d = i

                    if (n < distance):
                        n = distance
                        e = j
                        f = i

                    if (max < j):
                        max = j
                        g = j
                        h = i
        return [a,b],[g,h],[c,d],[e,f]



    def get_cloud(self):
        return self.cloud

    # def get_amplitudes(self):
    #     return self.amplitudes

    # def get_distances(self):
    #     return self.distances




a = matlab(1)
#a.ransac(100)
top = a.get_top_plane(0.11)
floor,n,d = a.get_floor_plane(0.06, top_plane=top)
[q,w],[e,r],[t,y],[u,i] = a.corner(top)

p3 = a.cloud[t,y]
x0,y0,z0  = p3
p1 = a.cloud[q,w]
x1,y1,z1= p1
p2 = a.cloud[e,r]
x2,y2,z2  = p2
p4 = a.cloud[u,i]
x3,y3,z3  = p4


height = np.abs(np.inner(p1, n) - d) / np.linalg.norm(n)
#height2 = np.abs(np.inner(p2, n) - d) / np.linalg.norm(n)
#height3 = np.abs(np.inner(p3, n) - d) / np.linalg.norm(n)
#height4 = np.abs(np.inner(p4, n) - d) / np.linalg.norm(n)
#height = (height1+height2+height3+height4)/4


length1 = np.sqrt((x3-x1)**2+(y3-y1)**2+(z3-z1)**2)
length2 = np.sqrt((x2-x0)**2+(y2-y0)**2+(z2-z0)**2)
length = (length1 + length2)/2

width1 = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
width2 = np.sqrt((x3-x2)**2+(y3-y2)**2+(z3-z2)**2)
width = (width1 + width2)/2

print(length)
print(width)
print(height)


#a.visualization(data_cloud=floor)
#a.visualization(data_cloud=top)

temp = top.astype(np.float32)
dst = cv2.cornerHarris(temp, 3, 7, 0.05)
dst = dst // (0.1 * dst.max()).astype(int)
dst = np.where(dst == 0, 0, 1)
#plt.imshow(dst)

display = top*100 + floor*180
plt.imshow(display)
plt.show()



'''
b = a.get_distances()
plt.imshow(b, cmap="gray")
plt.show()
'''


'''
disadvantage:
1.time
no upper bound on the time it takes to compute these parameters (except exhaustion).
2.iterations limitation
 When the number of iterations computed is limited the solution obtained may not be optimal, 
 and it may not even be one that fits the data in a good way.
'''






