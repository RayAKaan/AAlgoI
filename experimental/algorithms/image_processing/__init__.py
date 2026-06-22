
from typing import Any

import numpy as np

from aalgoi.algorithms.base import Algorithm


class GaussianBlur(Algorithm):
    name = "gaussian_blur"
    tags = ["image_processing", "blur", "noise_reduction", "fast"]
    complexity = {"time": "O(n)", "space": "O(n)"}
    patterns = ["Convolution", "LinearFilter"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "noise_reduction": {"score": 0.8, "conditions": {"noise_level": "high"}},
        "general": {"score": 0.7, "conditions": {}}
    }

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            from scipy import ndimage
            sigma = self.params.get("sigma", 1.0)
            return ndimage.gaussian_filter(data, sigma=sigma)
        return data

class MedianFilter(Algorithm):
    name = "median_filter"
    tags = ["image_processing", "denoise", "salt_pepper", "edge_preserving"]
    complexity = {"time": "O(n)", "space": "O(n)"}
    patterns = ["NonLinearFilter", "OrderStatistics"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "salt_pepper": {"score": 0.95, "conditions": {"noise_type": "salt_pepper"}},
        "general": {"score": 0.75, "conditions": {}}
    }

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            from scipy import ndimage
            size = self.params.get("filter_size", 3)
            return ndimage.median_filter(data, size=size)
        return data

class BilateralFilter(Algorithm):
    name = "bilateral_filter"
    tags = ["image_processing", "edge_preserving", "slow", "high_quality"]
    complexity = {"time": "O(n)", "space": "O(n)"}
    patterns = ["EdgePreserving", "NonLinearFilter"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "edge_preserving": {"score": 0.95, "conditions": {"preserve_edges": "true"}},
        "general": {"score": 0.6, "conditions": {}}
    }

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            try:
                from skimage.restoration import denoise_bilateral
                sigma_color = self.params.get("sigma_color", 0.1)
                sigma_spatial = self.params.get("sigma_spatial", 1.0)
                return denoise_bilateral(data, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
            except ImportError:
                from scipy import ndimage
                return ndimage.gaussian_filter(data, sigma=1.0)
        return data

class SobelEdgeDetection(Algorithm):
    name = "sobel_edge"
    tags = ["image_processing", "edge_detection", "fast"]
    complexity = {"time": "O(n)", "space": "O(n)"}
    patterns = ["Convolution", "GradientBased"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "edge_detection": {"score": 0.85, "conditions": {"task": "edge_detection"}},
        "general": {"score": 0.5, "conditions": {}}
    }

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            from scipy import ndimage
            dx = ndimage.sobel(data, axis=0)
            dy = ndimage.sobel(data, axis=1)
            return np.hypot(dx, dy)
        return data

class CLAHE(Algorithm):
    name = "clahe"
    tags = ["image_processing", "contrast", "enhancement", "adaptive"]
    complexity = {"time": "O(n)", "space": "O(n)"}
    patterns = ["AdaptiveHistogram", "ContrastEnhancement"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "low_contrast": {"score": 0.9, "conditions": {"contrast": "low"}},
        "general": {"score": 0.7, "conditions": {}}
    }

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            try:
                from skimage.exposure import equalize_adapthist
                clip_limit = self.params.get("clip_limit", 0.03)
                return equalize_adapthist(data, clip_limit=clip_limit)
            except ImportError:
                from skimage.exposure import equalize_hist
                return equalize_hist(data)
        return data


class CannyEdgeDetection(Algorithm):
    name = "canny_edge"
    tags = ["image_processing", "edge_detection", "multi_stage"]
    complexity = {"time": "O(n)", "space": "O(n)"}
    patterns = ["GradientBased", "HysteresisThresholding"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "edge_detection": {"score": 0.9, "conditions": {"task": "edge_detection"}},
        "general": {"score": 0.7, "conditions": {}}
    }

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            try:
                from skimage.feature import canny
                sigma = self.params.get("sigma", 1.0)
                return canny(data, sigma=sigma)
            except ImportError:
                from scipy import ndimage
                dx = ndimage.sobel(data, axis=0)
                dy = ndimage.sobel(data, axis=1)
                return np.hypot(dx, dy) > 0.3
        return data


class LaplacianEdgeDetection(Algorithm):
    name = "laplacian_edge"
    tags = ["image_processing", "edge_detection", "second_derivative"]
    complexity = {"time": "O(n)", "space": "O(n)"}
    patterns = ["Convolution", "GradientBased"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "edge_detection": {"score": 0.8, "conditions": {"task": "edge_detection"}},
        "general": {"score": 0.6, "conditions": {}}
    }

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            from scipy import ndimage
            return ndimage.laplace(data)
        return data


class NLMDenoising(Algorithm):
    name = "nlm_denoise"
    tags = ["image_processing", "denoise", "non_local_means", "high_quality"]
    complexity = {"time": "O(n log n)", "space": "O(n)"}
    patterns = ["NonLocalMeans", "PatchBased"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "denoise": {"score": 0.95, "conditions": {"noise_level": "high"}},
        "general": {"score": 0.7, "conditions": {}}
    }

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            try:
                from skimage.restoration import denoise_nl_means
                h = self.params.get("h", 0.1)
                return denoise_nl_means(data, h=h, fast_mode=True)
            except ImportError:
                from scipy import ndimage
                return ndimage.median_filter(data, size=3)
        return data


class MorphologyOperation(Algorithm):
    name = "morphology"
    tags = ["image_processing", "morphology", "binary"]
    complexity = {"time": "O(n)", "space": "O(n)"}
    patterns = ["StructuringElement", "SetTheory"]
    problem_types = ["IMAGE_PROCESSING"]
    performance_profiles = {
        "morphology": {"score": 0.9, "conditions": {"task": "morphology"}},
        "general": {"score": 0.5, "conditions": {}}
    }

    def __init__(self) -> None:
        super().__init__()
        self.params["operation"] = "dilate"

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray):
            if input_data.shape != output_data.shape:
                return False
        return True

    def process(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            from scipy import ndimage
            op = self.params.get("operation", "dilate")
            struct = self.params.get("structure", np.ones((3, 3)))
            operations = {
                "erode": ndimage.grey_erosion,
                "dilate": ndimage.grey_dilation,
                "open": ndimage.grey_opening,
                "close": ndimage.grey_closing,
            }
            fn = operations.get(op, ndimage.grey_dilation)
            return fn(data, structure=struct)
        return data
