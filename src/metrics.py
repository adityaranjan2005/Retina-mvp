import numpy as np
from typing import Dict, Tuple
from skimage.morphology import skeletonize
from scipy import ndimage
import cv2


def compute_skeleton_metrics(skeleton: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics from a binary skeleton.
    
    Args:
        skeleton: Binary skeleton array (0 or 1)
    
    Returns:
        Dictionary with metrics:
        - centerline_length_px: Total skeleton length in pixels
        - branch_points: Number of branch points
        - endpoints: Number of endpoints
        - tortuosity_proxy_mean: Mean tortuosity proxy
        - tortuosity_proxy_max: Max tortuosity proxy
    """
    skeleton = (skeleton > 0).astype(np.uint8)
    
    # Centerline length (number of skeleton pixels)
    centerline_length = np.sum(skeleton)
    
    # Find branch points and endpoints
    branch_points, endpoints = find_skeleton_junctions(skeleton)
    
    # Compute tortuosity proxy
    tortuosity_mean, tortuosity_max = compute_tortuosity_proxy(skeleton)
    
    return {
        'centerline_length_px': float(centerline_length),
        'branch_points': int(branch_points),
        'endpoints': int(endpoints),
        'tortuosity_proxy_mean': float(tortuosity_mean),
        'tortuosity_proxy_max': float(tortuosity_max)
    }


def find_skeleton_junctions(skeleton: np.ndarray) -> Tuple[int, int]:
    """
    Find branch points and endpoints in a skeleton.
    
    A branch point has 3+ neighbors, an endpoint has exactly 1 neighbor.
    
    Args:
        skeleton: Binary skeleton array
    
    Returns:
        (num_branch_points, num_endpoints)
    """
    # Create a 3x3 kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Count neighbors for each pixel
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    
    # Only consider skeleton pixels
    neighbor_count = neighbor_count * skeleton
    
    # Branch points: skeleton pixels with 3+ neighbors
    branch_points = np.sum(neighbor_count >= 3)
    
    # Endpoints: skeleton pixels with exactly 1 neighbor
    endpoints = np.sum(neighbor_count == 1)
    
    return branch_points, endpoints


def compute_tortuosity_proxy(skeleton: np.ndarray) -> Tuple[float, float]:
    """
    Compute a simple tortuosity proxy based on path analysis.
    
    Tortuosity proxy = actual_path_length / euclidean_distance
    
    For each connected component, compute the ratio of skeleton length to 
    end-to-end distance. Returns mean and max across all components.
    
    Args:
        skeleton: Binary skeleton array
    
    Returns:
        (mean_tortuosity, max_tortuosity)
    """
    skeleton = (skeleton > 0).astype(np.uint8)
    
    # Label connected components
    labeled, num_components = ndimage.label(skeleton)
    
    if num_components == 0:
        return 0.0, 0.0
    
    tortuosities = []
    
    for label_id in range(1, num_components + 1):
        component = (labeled == label_id).astype(np.uint8)
        
        # Get coordinates of this component
        coords = np.argwhere(component > 0)
        
        if len(coords) < 2:
            continue
        
        # Path length is the number of pixels
        path_length = len(coords)
        
        # Euclidean distance between furthest points
        # Use the first and last point as approximation (or find actual endpoints)
        endpoints_coords = find_component_endpoints(component)
        
        if len(endpoints_coords) >= 2:
            # Use the two furthest endpoints
            p1, p2 = endpoints_coords[0], endpoints_coords[-1]
            euclidean_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        else:
            # Fallback: use bounding box diagonal
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
    """
    Find endpoints of a connected component.
    
    Args:
        component: Binary array of a single connected component
    
    Returns:
        List of endpoint coordinates
    """
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = ndimage.convolve(component.astype(np.uint8), kernel, mode='constant')
    neighbor_count = neighbor_count * component
    
    # Endpoints have exactly 1 neighbor
    endpoints = np.argwhere(neighbor_count == 1)
    
    return endpoints.tolist() if len(endpoints) > 0 else []


def extract_centerline_from_vessel(vessel_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Extract centerline from vessel prediction.
    
    Args:
        vessel_mask: Binary or probability vessel mask
        threshold: Threshold for binarization
    
    Returns:
        Binary centerline mask
    """
    # Binarize if needed
    if vessel_mask.max() <= 1.0:
        binary_vessel = (vessel_mask > threshold).astype(bool)
    else:
        binary_vessel = (vessel_mask > threshold * 255).astype(bool)
    
    # Skeletonize
    skeleton = skeletonize(binary_vessel)
    
    return skeleton.astype(np.uint8)


def postprocess_vessel_mask(vessel_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Postprocess vessel mask with morphological closing to bridge gaps.
    
    Args:
        vessel_mask: Binary vessel mask
        kernel_size: Size of morphological kernel
    
    Returns:
        Postprocessed vessel mask
    """
    vessel_mask = (vessel_mask > 0).astype(np.uint8) * 255
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, kernel)
    
    return closed
