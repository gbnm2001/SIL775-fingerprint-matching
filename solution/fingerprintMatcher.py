import numpy as np
import cv2
import math
import fingerprint_enhancer as fpe
import fingerprint_feature_extractor as ffx

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
        g_thresh /= 10

        for i in range(0, f_img.shape[0], block_size):
            for j in range(0, f_img.shape[1], block_size):
                x = min(f_img.shape[0], i+block_size)
                y = min(f_img.shape[1], j+block_size)
                # local grayscale variance
                # set the pixels to zero if their value is less than g_thresh
                if np.var(f_img[i:x, j: y]) <= g_thresh:
                    seg_mask[i:x, j:y] = 0
        return seg_mask

    def segmentation_masking(self, f_img, seg_mask,block_size=15):
        seg_img = f_img.copy()
        # kernel open close
        koc = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size*2, block_size*2))
        # opening and closing erosion and dilation
        seg_mask = cv2.erode(seg_mask, koc, iterations=1)
        seg_mask = cv2.dilate(seg_mask, koc, iterations=1)
        seg_mask = cv2.dilate(seg_mask, koc, iterations=1)
        seg_mask = cv2.erode(seg_mask, koc, iterations=1)
        for i in range(seg_mask.shape[0]):
            for j in range(seg_mask.shape[1]):
                if(seg_mask[i][j] == 0):
                    seg_img[i][j]=255
        return seg_img

    def normalize_img(self, seg_img):
        # desired mean and desired variance
        dm, dv = 128.0, 7500.0
        # current mean and current variance
        cm, cv = np.mean(seg_img), np.var(seg_img)
        norm_img = np.empty([seg_img.shape[0], seg_img.shape[1]], float)
        for i in  range(seg_img.shape[0]):
            for j in range(seg_img.shape[1]):
                n_val = (math.sqrt(math.pow(seg_img[i][j]-cm,2)*(dv/cv)))
                if seg_img[i][j] > cm:
                    norm_img[i][j] = dm + n_val
                else:
                    norm_img[i][j] = dm - n_val
        return norm_img

    def preprocess_img(self, f_path):
        # returns 
        f_img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        # perform segmentation on the image
        seg_mask = self.segment_img(f_img)
        seg_img = self.segmentation_masking(f_img, seg_mask)
        norm_img = self.normalize_img(seg_img)
        enhanced_img = fpe.enhance_Fingerprint(norm_img)
        FeaturesTerminations, FeaturesBifurcations = ffx.extract_minutiae_features(enhanced_img, spuriousMinutiaeThresh=10, invertImage = False, showResult=False, saveResult = True)
        extracted_minutiae_terminations = map(lambda minutiae: [minutiae.locX, minutiae.locY, minutiae.Orientation] , FeaturesTerminations)
        extracted_minutiae_bifurcations = map(lambda minutiae: [minutiae.locX, minutiae.locY, minutiae.Orientation], FeaturesBifurcations)
        return extracted_minutiae_bifurcations, extracted_minutiae_terminations

    def execute_matcher(self):
        ex_minb1, ex_mint1 = self.preprocess_img(f_path=self.img1_path)
        ex_minb2, ex_mint2 = self.preprocess_img(f_path=self.img2_path)


fm = FingerprintMatcher('101', '/home/kshitij86/Desktop/iitd/course/sem2/biometric/assignments/assignment1/SIL775-fingerprint-matching/solution/testcases2')
fm.execute_matcher()
