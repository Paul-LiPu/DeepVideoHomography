
import cv2
import numpy as np

def sift_ransac_func():
    def sift_ransac(match_img, ref_img, num_choosing=None, logger=None, return_kp=False):
        """
        Calculate homography by SIFT + RANSAC.
        :param match_img: target image to be transformed.
        :param ref_img: reference image.
        :param num_choosing: Number of matched points used for RANSAC.
        :return: Homography matrix H. It is able to tranform points in match_img to corresponding points in ref_img.
        p_ref = H * p_match.
        """
        min_match_count = 4
        sift_obj = cv2.xfeatures2d.SIFT_create()  # SIFT
        kp1, des1 = sift_obj.detectAndCompute(match_img, None)
        kp2, des2 = sift_obj.detectAndCompute(ref_img, None)

        # Brute force matcher.
        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
        except:
            logger.info('Match error, return identity matrix')
            return np.eye(3)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < min_match_count:
          logger.info('===> Not enough matched features. Return identity matrix')
          return np.eye(3)

        if num_choosing is None:
            num_choosing = len(matches)
        if num_choosing > len(matches):
            num_choosing = len(matches)

        goodMatches = matches[0:num_choosing]
        srcPoints = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(srcPoints, dstPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if not return_kp:
            return H
        else:
            return H, goodMatches, mask
    return sift_ransac


def orb_ransac_func(WTA_K=2):
    def orb_ransac(match_img, ref_img, num_choosing=None, logger=None):
        """
        Calculate homography by ORB + RANSAC.
        :param match_img: target image to be transformed.
        :param ref_img: reference image.
        :param num_choosing: Number of matched points used for RANSAC.
        :param WTA_K: parameter for ORB.
        :return: Homography matrix H. It is able to tranform points in match_img to corresponding points in ref_img.
        p_ref = H * p_match.
        """
        min_match_count = 4
        orb_obj = cv2.ORB_create(WTA_K=WTA_K)  # ORB
        kp1, des1 = orb_obj.detectAndCompute(match_img, None)
        kp2, des2 = orb_obj.detectAndCompute(ref_img, None)

        # Brute force matcher.
        if WTA_K <= 2:
            bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING2, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
        except:
            logger.info('Match error, return identity matrix')
            return np.eye(3)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < min_match_count:
          logger.info('===> Not enough matched features. Return identity matrix')
          return np.eye(3)

        if num_choosing is None:
            num_choosing = len(matches)
        if num_choosing > len(matches):
            num_choosing = len(matches)

        goodMatches = matches[0:num_choosing]
        srcPoints = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        H, __ = cv2.findHomography(srcPoints, dstPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        return H
    return orb_ransac

def ecc_fuc(num_iterations=100):
    def ecc(match_img, ref_img, logger=None):
        # Motion model
        warp_mode = cv2.MOTION_HOMOGRAPHY

        # Define 3x3 matrice and initialize  the matrix to identity
        warp_matrix = np.eye(3, 3, dtype=np.float32)

        # Number of iteration
        num_iters = num_iterations

        # Threshold of the increment in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iters, termination_eps)
        try:
            (cc, H) = cv2.findTransformECC(match_img, ref_img, warp_matrix, warp_mode, criteria)
        except:
            logger.info("Error in convergence...")
            return np.eye(3)
        return H
    return ecc

def identity_func():
    def identity(match_img, ref_img, logger=None):
        return np.eye(3)
    return identity
