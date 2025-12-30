import numpy as np
from typing import Dict, Tuple
from skimage.morphology import skeletonize
from scipy import ndimage
import cv2


def compute_skeleton_metrics(skeleton: np.ndarray) -> Dict[str, float]:
    skeleton = (skeleton > 0).astype(np.uint8)
    centerline_length = np.sum(skeleton)
    branch_points, endpoints = find_skeleton_junctions(skeleton)
    tortuosity_mean, tortuosity_max = compute_tortuosity_proxy(skeleton)
    
    return {
        'centerline_length_px': float(centerline_length),
        'branch_points': int(branch_points),
        'endpoints': int(endpoints),
        'tortuosity_proxy_mean': float(tortuosity_mean),
        'tortuosity_proxy_max': float(tortuosity_max)
    }


def find_skeleton_junctions(skeleton: np.ndarray) -> Tuple[int, int]:
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    neighbor_count = neighbor_count * skeleton
    branch_points = np.sum(neighbor_count >= 3)
    endpoints = np.sum(neighbor_count == 1)
    return branch_points, endpoints


def compute_tortuosity_proxy(skeleton: np.ndarray) -> Tuple[float, float]:
    skeleton = (skeleton > 0).astype(np.uint8)
    labeled, num_components = ndimage.label(skeleton)
    
    if num_components == 0:
        return 0.0, 0.0
    
    tortuosities = []
    
    for label_id in range(1, num_components + 1):
        component = (labeled == label_id).astype(np.uint8)
        coords = np.argwhere(component > 0)
        
        if len(coords) < 2:
            continue
        
        path_length = len(coords)
        endpoints_coords = find_component_endpoints(component)
        
        if len(endpoints_coords) >= 2:
            p1, p2 = endpoints_coords[0], endpoints_coords[-1]
            euclidean_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        else:
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            euclidean_dist = np.sqrt(np.sum((max_coords - min_coords)**2))
        
        if euclidean_dist > 0:
            tortuosity = path_length / euclidean_dist
            tortuosities.append(tortuosity)
    
    if len(tortuosities) == 0:
        return 0.0, 0.0
    
    return float(np.mean(tortuosities)), float(np.max(tortuosities))


def find_component_endpoints(component: np.ndarray) -> list:
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = ndimage.convolve(component.astype(np.uint8), kernel, mode='constant')
    neighbor_count = neighbor_count * component
    endpoints = np.argwhere(neighbor_count == 1)
    return endpoints.tolist() if len(endpoints) > 0 else []


def extract_centerline_from_vessel(vessel_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if vessel_mask.max() <= 1.0:
        binary_vessel = (vessel_mask > threshold).astype(bool)
    else:
        binary_vessel = (vessel_mask > threshold * 255).astype(bool)
    skeleton = skeletonize(binary_vessel)
    return skeleton.astype(np.uint8)


def postprocess_vessel_mask(vessel_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    vessel_mask = (vessel_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, kernel)
    return closed
