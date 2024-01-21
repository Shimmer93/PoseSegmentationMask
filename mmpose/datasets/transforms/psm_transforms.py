from mmpose.datasets.datasets.utils import parse_pose_metainfo

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of
from scipy.ndimage import gaussian_filter

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix

def draw_line(canvas, start, end, value=1, overlength=0):
    # canvas: torch.Tensor of shape (height, width)
    # start: torch.Tensor of shape (2,)
    # end: torch.Tensor of shape (2,)

    # Bresenham's line algorithm
    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    h, w = canvas.size(0), canvas.size(1)
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    if overlength > 0:
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            x0 = int(np.clip(x0 - dx * overlength, 0, h-1))
            y0 = int(np.clip(y0 - dy * overlength, 0, w-1))
            x1 = int(np.clip(x1 + dx * overlength, 0, h-1))
            y1 = int(np.clip(y1 + dy * overlength, 0, w-1))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 >= 0 and x0 < w and y0 >= 0 and y0 < h:
        canvas[y0, x0] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def skeleton_to_body_mask(skl, limbs, height, width):
    mask = np.zeros((height, width))
    for i, pair in enumerate(limbs.items()):
        start = skl[pair[0]][:2]
        end = skl[pair[1]][:2]
        draw_line(mask, start, end, value=1, overlength=0.2)
    mask = gaussian_filter(mask, sigma=3, radius=1)
    mask = (mask > 0).float()
    return mask

def skeleton_to_joint_mask(skl, height, width):
    mask = np.zeros((skl.shape[0], height, width))
    for i, pt in enumerate(skl):
        canvas = np.zeros((height, width))
        canvas[int(pt[1]), int(pt[0])] = 1
        canvas = gaussian_filter(canvas, sigma=3, radius=1)
        canvas = (canvas > 0).float()
        mask[i] = canvas
    return mask

@TRANSFORMS.register_module()
class GenerateRoughBodyAndJointMasks(BaseTransform):
    
    def __init__(self,
                 input_size: Tuple[int, int]) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        