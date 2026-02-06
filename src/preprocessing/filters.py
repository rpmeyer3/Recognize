from __future__ import annotations
import random
import cv2
import numpy as np
import torch


class PreprocessingPipeline:
    def __init__(self, cfg, stochastic=True, apply_prob=0.3):
        self.stochastic = stochastic
        self.apply_prob = apply_prob
        bil = cfg.get("bilateral", {})
        self.bil_d = bil.get("d", 9)
        self.bil_sc = bil.get("sigma_color", 75)
        self.bil_ss = bil.get("sigma_space", 75)
        nlm = cfg.get("nlm", {})
        self.nlm_h = nlm.get("h", 10)
        self.nlm_tw = nlm.get("template_window", 7)
        self.nlm_sw = nlm.get("search_window", 21)

    def bilateral_filter(self, img):
        u8 = (img * 255).clip(0, 255).astype(np.uint8)
        return cv2.bilateralFilter(u8, self.bil_d, self.bil_sc, self.bil_ss).astype(np.float32) / 255.0

    def nlm_filter(self, img):
        u8 = (img * 255).clip(0, 255).astype(np.uint8)
        return cv2.fastNlMeansDenoising(u8, None, h=self.nlm_h,
            templateWindowSize=self.nlm_tw, searchWindowSize=self.nlm_sw).astype(np.float32) / 255.0

    def __call__(self, image):
        squeeze = image.dim() == 3
        img_np = image.squeeze(0).cpu().numpy() if squeeze else image.cpu().numpy()
        if self.stochastic:
            if random.random() < self.apply_prob:
                img_np = self.bilateral_filter(img_np)
            if random.random() < self.apply_prob * 0.5:
                img_np = self.nlm_filter(img_np)
        else:
            img_np = self.bilateral_filter(img_np)
        result = torch.from_numpy(img_np)
        if squeeze:
            result = result.unsqueeze(0)
        return result.to(image.device)
