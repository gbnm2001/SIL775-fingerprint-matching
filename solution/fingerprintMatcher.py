import numpy as np
import cv2

class FingerprintMatcher(object):
    def __init__(self, img_no, testcases_dir):
        self.testcases_dir = f'{testcases_dir}'
        self.img1_path = f'{self.testcases_dir}/{img_no}_1.tif'
        self.img2_path = f'{self.testcases_dir}/{img_no}_2.tif'

    def segment_img(self, f_img, block_size=15):
        
        # get the segmentation mask
        seg_mask = np.ones(f_img.shape)

        # global threshold
        g_thresh = np.var(f_img, None)
        g_thresh *= 0.1

        r = f_img.shape[0]
        c = f_img.shape[1]
        for i in range(0, r, block_size):
            for j in range(c):
                x = min(r, i+block_size)
                y = min(c, j+block_size)
                # local grayscale variance
                # set the pixels to zero if their value is less than g_thresh
                if np.var(f_img[i:x, j: y]) <= g_thresh:
                    seg_mask[i:x, j:y] = 0
        return seg_mask


    def preprocess_img(self, f_path):
        f_img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)

        # perform segmentation on the image
        return self.segment_img(f_img)

    def execute_matcher(self):
        print(self.preprocess_img(f_path=self.img1_path))
        print(self.preprocess_img(f_path=self.img2_path))

fm = FingerprintMatcher('101', '/home/kshitij86/Desktop/iitd/course/sem2/biometric/assignments/assignment1/SIL775-fingerprint-matching/solution/testcases')
fm.execute_matcher()