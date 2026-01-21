"""
SIFT and LoFTR
"""
from typing import Any, List, Tuple

import torch
from PIL import Image
import numpy as np
import cv2
import kornia as Kn
import kornia.feature as KnF

class CVModel:
    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg

    def preprocess(self, images: List[Image.Image]) -> List[Any]:
        raise NotImplementedError

    def get_matches(self, images: List[Any]) -> List[np.ndarray]:
        raise NotImplementedError

    def cal_relative_pose(self, mp1: np.ndarray, mp2: np.ndarray, K: np.ndarray) -> Any:
        """
        Use essential matrix to compute relative pose
        """
        try: 
            E, _ = cv2.findEssentialMat(mp1, mp2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, mp1, mp2, K)
        except Exception as e:
            print(f"Error in recoverPose: {e}Note that, mp1.shape: {mp1.shape}, mp2.shape: {mp2.shape}. Maybe not enough matches?")
            R, t = None, None
        return R, t # in 2d array

    def pipeline(self, images: List[Image.Image], K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        processed_images = self.preprocess(images)
        mp1, mp2 = self.get_matches(processed_images)
        R, t = self.cal_relative_pose(mp1, mp2, K)
        return R, t # in 2d array
    

class SIFT(CVModel):
    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)
        # self.model = cv2.SIFT_create()
        self.model = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)

    def preprocess(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Make PIL images to gray and np.ndarray
        """
        processed_images = []
        for img in images:
            # img = img.resize((640, 480))  # Resize
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY) # H x W
            processed_images.append(gray)
        return processed_images

    def get_matches(self, images: List[np.ndarray]) -> List:
        """
        Get SIFT keypoints and matches between two images with FLANN
        """
        img2, img1 = images # tgt image is img1, src image is img2
        kp1, des1 = self.model.detectAndCompute(img1, None)
        kp2, des2 = self.model.detectAndCompute(img2, None)

        index_params = dict(algorithm=1, trees=4)
        # search_params = dict(checks=50)
        search_params = dict(checks=200)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        mp1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        mp2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return mp1, mp2


class LoFTR(CVModel):
    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pretrained="indoor" for http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt
        # pretrained="indoor_new" for http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt
        self.model = KnF.LoFTR(pretrained="indoor_new").to(self.device) 

    def preprocess(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Make PIL images to gray and torch.Tensor
        """
        processed_images = []
        for img in images:
            img = img.resize((640, 480))  # LoFTR default input size
            img = np.array(img) # H x W x 3
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # H x W
            tensor_img = Kn.image_to_tensor(gray, False).float() / 255.0  # 1 x 1 x H x W
            processed_images.append(tensor_img.squeeze(0).to(self.device)) # 1 x H x W
        return processed_images
    
    def get_matches(self, images: List[torch.Tensor]) -> List:
        """
        Get LoFTR keypoints and matches between two images
        """
        img2, img1 = images  # tgt image is img1, src image is img2
        batch = {"image0": img1.unsqueeze(0), "image1": img2.unsqueeze(0)}
        with torch.no_grad():
            matches = self.model(batch)

        mkpts0 = matches["keypoints0"].cpu().numpy()
        mkpts1 = matches["keypoints1"].cpu().numpy()
        return mkpts0, mkpts1
