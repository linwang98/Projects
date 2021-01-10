import cv2
from mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt

# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
# FaceDetector类提供用于检测，跟踪和对齐面部的方法。
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    # 准备面部检测器； 指定用于检测，跟踪和对齐的所有参数。
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.模板匹配的参考（初始人脸检测）
        self.reference = None

        # Size of face image after landmark-based alignment.基于界标对齐后的人脸图像大小。
        self.aligned_image_size = aligned_image_size

        # ToDo: Specify all parameters for template matching.指定用于模板匹配的所有参数。
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

        # ToDo: Track a face in a new image using template matching.

    def track_face(self, image):
        self.detect_face(image)
        image3 = np.copy(image)
        method = cv2.TM_CCOEFF_NORMED
        # re - initialization
        # Store the coordinates of matched area in a numpy array
        while True:

            # Re-initialize tracker
            template = self.reference
            # plt.imshow(template, cmap='gray')
            # plt.show()
            # w = self.aligned_image_size//2
            # h = self.aligned_image_size//2
            win =  self.tm_window_size
            w = self.aligned_image_size
            h = self.aligned_image_size
            res = cv2.matchTemplate(image, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            top_left = max_loc
            bottom_right = (max_loc[0]+w, max_loc[1]+h)

            # top_left_x = max(max_loc[0] - w, 0)
            # top_left_y = max(max_loc[1] - h, 0)
            # bottom_right_x = min(max_loc[0] + w - 1, image.shape[0] - 1)
            # bottom_right_y = min(max_loc[1] + h - 1, image.shape[1] - 1)
            # top_left = (top_left_x, top_left_y)
            # bottom_right = (bottom_right_x, bottom_right_y)
            cv2.rectangle(image, top_left, bottom_right, 255, 2)
            # image2 = image3[ top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]
            image2 = image3[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0],:]
            face_rect = [max_loc[0],max_loc[1],w,h]

            # plt.subplot(121)
            # plt.imshow(res, cmap='gray')
            # plt.subplot(122)
            # plt.imshow(image, cmap='gray')
            # plt.show()
            # plt.imshow(image2, cmap='gray')
            # plt.show()
            if max_val < self.tm_threshold:

                self.detect_face(image2)
                # self.detect_face(image)

            if max_val > self.tm_threshold:
                break

        return {"rect":face_rect , "image": image2, "aligned": self.align_face(image,face_rect), "response": 0}
    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.检索给定图像中的所有可检测面部。
        detections = self.detector.detect_faces(image)
        # print(detections)
        if not detections:
            self.reference = None
            return None

        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]
        # print(face_rect) [159, 59, 67, 76]
        # x, y, width, height = face_rect
        detect = self.crop_face(image, face_rect)
        self.reference = detect
        aligned = self.align_face(image, face_rect)

        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}



    # Face alignment to predefined size(224×224 pixel）
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.根据检测到的边界框裁剪脸部。
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

# img = plt.imread("/Users/rolin/Desktop/Projcv/exe04/supplementary_material/datasets/training_data/Nancy_Sinatra/0001.jpg")
# face = FaceDetector()
# # m = face.detect_face(img)
# n = face.track_face(img)
# # print(n)
# # m = face.track_face(img2)
# #
