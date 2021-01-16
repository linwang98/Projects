import cv2
from mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

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

        image3 = np.copy(image)
        method = cv2.TM_CCOEFF_NORMED
        win = self.tm_window_size

        # Re-initialize tracker
        if self.reference is None:
            self.reference = self.detect_face(image)
            return self.reference
        # template = self.reference["crop"]
        x, y, width, height = self.reference["rect"]
        template = self.crop_face(self.reference["image"], self.reference["rect"])
        # face_rect_n = [x, y, width+win, height+win]
        # img = self.crop_face(image, face_rect_n)
        res = cv2.matchTemplate(image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        face_rect = [ max_loc[0], max_loc[1], width, height ]
        face_align = self.align_face(image, face_rect)
        # cv2.rectangle(image,
        #               (face_rect[0], face_rect[1]),
        #               (face_rect[0] + face_rect[2] - 1, face_rect[1] + face_rect[3] - 1), (0, 255, 0), 2)
        # cv2.rectangle(image, top_left, bottom_right, 255, 2)
        #
        # plt.subplot(121)
        # plt.imshow(res)
        # plt.subplot(122)
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(template)
        # plt.show()
        if max_val < self.tm_threshold:
            self.detect_face(image)
        else:
            self.reference["aligned"] = face_align
            self.reference["rect"] = face_rect
            self.reference["image"] = image

        return self.reference

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

    def load_images(self,folder):
        images = []
        for filename in os.listdir(folder):
            img = mpimg.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        return images

# face = FaceDetector()
# folder = "/Users/rolin/Desktop/Projcv/exe04/supplementary_material/datasets/training_data/Manuel_Pellegrini"
# m = face.load_images(folder)
# for i in range(len(m)):
#     # m = face.detect_face(m[i])
#     n = face.track_face(m[i])
