"""Simulates handheld / low-cost ultrasound degradation on clean images."""
import random
import numpy as np
import cv2


class DegradationPipeline:
    """
    Applies a configurable chain of ultrasound-realistic degradations to a
    numpy uint8 image (H, W, 3) and returns the degraded image in the same
    format.

    All parameter ranges are drawn uniformly at call time so each image gets
    a different degradation realisation — important for unpaired training.
    """

    def __init__(
        self,
        speckle_sigma_range=(0.05, 0.30),
        gaussian_sigma_range=(5.0, 25.0),
        downsample_factor_range=(2, 4),
        blur_kernel_range=(3, 7),
        p_speckle=0.8,
        p_gaussian=0.5,
        p_downsample=0.7,
        p_blur=0.4,
    ):
        self.speckle_sigma_range = speckle_sigma_range
        self.gaussian_sigma_range = gaussian_sigma_range
        self.downsample_factor_range = downsample_factor_range
        self.blur_kernel_range = blur_kernel_range
        self.p_speckle = p_speckle
        self.p_gaussian = p_gaussian
        self.p_downsample = p_downsample
        self.p_blur = p_blur

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape[:2]

        if random.random() < self.p_speckle:
            img = self._add_speckle(img)

        if random.random() < self.p_gaussian:
            img = self._add_gaussian(img)

        if random.random() < self.p_downsample:
            img = self._downsample(img, h, w)

        if random.random() < self.p_blur:
            img = self._motion_blur(img)

        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def _add_speckle(self, img: np.ndarray) -> np.ndarray:
        sigma = random.uniform(*self.speckle_sigma_range)
        # Multiplicative Gamma-distributed noise (standard speckle model)
        noise = np.random.gamma(shape=1.0 / (sigma ** 2), scale=sigma ** 2, size=img.shape).astype(np.float32)
        return img * noise

    def _add_gaussian(self, img: np.ndarray) -> np.ndarray:
        sigma = random.uniform(*self.gaussian_sigma_range) / 255.0
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        return img + noise

    def _downsample(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        factor = random.randint(*self.downsample_factor_range)
        small_h, small_w = max(1, h // factor), max(1, w // factor)
        downsampled = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_NEAREST)

    def _motion_blur(self, img: np.ndarray) -> np.ndarray:
        k = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
        kernel = np.zeros((k, k), dtype=np.float32)
        if random.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k   # horizontal
        else:
            kernel[:, k // 2] = 1.0 / k   # vertical
        return cv2.filter2D(img, -1, kernel)
