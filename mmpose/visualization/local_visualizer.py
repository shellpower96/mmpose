# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from PIL import Image, ImageDraw, ImageFont
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample
from .opencv_backend_visualizer import OpencvBackendVisualizer
from .simcc_vis import SimCCVisualizer
from PIL import Image

def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


@VISUALIZERS.register_module()
class PoseLocalVisualizer(OpencvBackendVisualizer):
    """MMPose Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to ``None``
        vis_backends (list, optional): Visual backend config list. Defaults to
            ``None``
        save_dir (str, optional): Save file dir for all storage backends.
            If it is ``None``, the backend storage will not save any data.
            Defaults to ``None``
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to ``'green'``
        kpt_color (str, tuple(tuple(int)), optional): Color of keypoints.
            The tuple of color should be in BGR order. Defaults to ``'red'``
        link_color (str, tuple(tuple(int)), optional): Color of skeleton.
            The tuple of color should be in BGR order. Defaults to ``None``
        line_width (int, float): The width of lines. Defaults to 1
        radius (int, float): The radius of keypoints. Defaults to 4
        show_keypoint_weight (bool): Whether to adjust the transparency
            of keypoints according to their score. Defaults to ``False``
        alpha (int, float): The transparency of bboxes. Defaults to ``1.0``

    Examples:
        >>> import numpy as np
        >>> from mmengine.structures import InstanceData
        >>> from mmpose.structures import PoseDataSample
        >>> from mmpose.visualization import PoseLocalVisualizer

        >>> pose_local_visualizer = PoseLocalVisualizer(radius=1)
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                          [8, 8]]])
        >>> gt_pose_data_sample = PoseDataSample()
        >>> gt_pose_data_sample.gt_instances = gt_instances
        >>> dataset_meta = {'skeleton_links': [[0, 1], [1, 2], [2, 3]]}
        >>> pose_local_visualizer.set_dataset_meta(dataset_meta)
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample)
        >>> pose_local_visualizer.add_datasample(
        ...                       'image', image, gt_pose_data_sample,
        ...                        out_file='out_file.jpg')
        >>> pose_local_visualizer.add_datasample(
        ...                        'image', image, gt_pose_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                       [8, 8]]])
        >>> pred_instances.score = np.array([0.8, 1, 0.9, 1])
        >>> pred_pose_data_sample = PoseDataSample()
        >>> pred_pose_data_sample.pred_instances = pred_instances
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample,
        ...                         pred_pose_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = 'green',
                 kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = 'red',
                 link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (255, 255, 255),
                 skeleton: Optional[Union[List, Tuple]] = None,
                 line_width: Union[int, float] = 1,
                 radius: Union[int, float] = 3,
                 show_keypoint_weight: bool = False,
                 backend: str = 'opencv',
                 alpha: float = 1.0,
                 side: int = 1,):

        warnings.filterwarnings(
            'ignore',
            message='.*please provide the `save_dir` argument.*',
            category=UserWarning)

        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            backend=backend)

        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.skeleton = skeleton
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight
        self.side = side
        # Set default value. When calling
        # `PoseLocalVisualizer().set_dataset_meta(xxx)`,
        # it will override the default value.
        self.dataset_meta = {}

    def set_dataset_meta(self,
                         dataset_meta: Dict,
                         skeleton_style: str = 'mmpose', side: int = 1, show_lines: bool = True):
        """Assign dataset_meta to the visualizer. The default visualization
        settings will be overridden.

        Args:
            dataset_meta (dict): meta information of dataset.
        """
        self.side = side
        self.show_lines = show_lines
        if skeleton_style == 'openpose':
            dataset_name = dataset_meta['dataset_name']
            if dataset_name == 'coco':
                dataset_meta = parse_pose_metainfo(
                    dict(from_file='configs/_base_/datasets/coco_openpose.py'))
            elif dataset_name == 'coco_wholebody':
                dataset_meta = parse_pose_metainfo(
                    dict(from_file='configs/_base_/datasets/'
                         'coco_wholebody_openpose.py'))
            else:
                raise NotImplementedError(
                    f'openpose style has not been '
                    f'supported for {dataset_name} dataset')

        if isinstance(dataset_meta, dict):
            self.dataset_meta = dataset_meta.copy()
            self.bbox_color = dataset_meta.get('bbox_color', self.bbox_color)
            self.kpt_color = dataset_meta.get('keypoint_colors',
                                              self.kpt_color)
            self.link_color = dataset_meta.get('skeleton_link_colors',
                                               self.link_color)
            self.skeleton = dataset_meta.get('skeleton_links', self.skeleton)
        # sometimes self.dataset_meta is manually set, which might be None.
        # it should be converted to a dict at these times
        if self.dataset_meta is None:
            self.dataset_meta = {}
    def _draw_instances_bbox(self, image: np.ndarray,
                             instances: InstanceData) -> np.ndarray:
        """Draw bounding boxes and corresponding labels of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            self.draw_bboxes(
                bboxes,
                edge_colors=self.bbox_color,
                alpha=self.alpha,
                line_widths=self.line_width)
        else:
            return self.get_image()

        if 'labels' in instances and self.text_color is not None:
            classes = self.dataset_meta.get('classes', None)
            labels = instances.labels

            positions = bboxes[:, :2]
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'

                if isinstance(self.bbox_color,
                              tuple) and max(self.bbox_color) > 1:
                    facecolor = [c / 255.0 for c in self.bbox_color]
                else:
                    facecolor = self.bbox_color

                self.draw_texts(
                    label_text,
                    pos,
                    colors=self.text_color,
                    font_sizes=int(13 * scales[i]),
                    vertical_alignments='bottom',
                    bboxes=[{
                        'facecolor': facecolor,
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    def my_draw_instances_kpts(self,
                             image: np.ndarray,
                             instances: InstanceData,
                             kpt_thr: float = 0.3,
                             show_kpt_idx: bool = False,
                             skeleton_style: str = 'mmpose',
                             side: int = 1,):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        ## 定义点和线的颜色
        img_h, img_w, _ = image.shape
        transparent_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        self.set_image(transparent_image)
        
        if instances.keypoints.shape[1] ==26:
            a = np.array([255,69,0], dtype=np.uint8)
            self.kpt_color = np.expand_dims(a,0).repeat(26,axis=0)
            b = np.array([51, 153, 255], dtype=np.uint8)
            self.link_color = np.expand_dims(b,0).repeat(27,axis=0)
            if skeleton_style == 'openpose':
                return self._draw_instances_kpts_openpose(image, instances,
                                                        kpt_thr)

            
            ###########################################
            print(self.side)
            if self.side == 1:
                #####正面
                whether_visible = np.ones((1,26))
                whether_visible[0,24] = 0
                whether_visible[0,25] = 0
                #####侧面左
            elif self.side == 2:
                whether_visible = np.array([[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
                #####侧面右
            elif self.side == 3:
                whether_visible = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
                #####背面
            elif self.side == 4:
                whether_visible = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]])
        
        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            for kpts, visible, whe in zip(keypoints, keypoints_visible, whether_visible):
                kpts = np.array(kpts, copy=False)
            ### 点改颜色
                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')
                #link 改颜色
                # draw links
                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')
                    if kpts.shape[0] ==26:
                        if self.side == 1:
                            self.skeleton = [(15, 13), (13, 11), (11, 19), (16, 14), (14, 12), (12, 19), (17, 18), (18, 19), (18, 5), (5, 7), (7, 9), (18, 6), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (15, 20), (15, 22), (16, 21), (16, 23)]
                        if self.side == 2:
                            self.skeleton = [(3,5), (5,11), (11,13), (13,15), (15,24), (15,20)]
                        if self.side == 3:
                            self.skeleton = [(4,6), (6,12), (12,14), (14,16), (16,25), (16,21)]
                        if self.side == 4:
                            self.skeleton = [(15, 13), (13, 11), (11, 19), (16, 14), (14, 12), (12, 19), (17, 18), (18, 19), (18, 5), (5, 7), (7, 9), (18, 6), (6, 8), (8, 10), (15,24), (16, 25)]
                    
                    for sk_id, sk in enumerate(self.skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h or visible[sk[0]] < kpt_thr
                                or visible[sk[1]] < kpt_thr
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue
                        #
                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            #######################################################改固定颜色
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0,
                                min(1,
                                    0.5 * (visible[sk[0]] + visible[sk[1]])))

                        self.draw_lines(
                            X, Y, color, line_widths=self.line_width)
                #点改颜色
                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if visible[kid] < kpt_thr or kpt_color[kid] is None or whe[kid] == 0:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    ###################################################改固定颜色
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, visible[kid]))
                    self.draw_circles(
                        kpt,
                        radius=np.array([self.radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=self.radius)
                    if show_kpt_idx:
                        kpt_idx_coords = kpt + [self.radius, -self.radius]
                        self.draw_texts(
                            str(kid),
                            kpt_idx_coords,
                            colors=color,
                            font_sizes=self.radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')
        img = self.get_image()
        cv2.imwrite('demo/resources/skeleton.png',img)
    def _draw_instances_bbox(self, image: np.ndarray,
                             instances: InstanceData) -> np.ndarray:
        """Draw bounding boxes and corresponding labels of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            self.draw_bboxes(
                bboxes,
                edge_colors=self.bbox_color,
                alpha=self.alpha,
                line_widths=self.line_width)
        else:
            return self.get_image()

        if 'labels' in instances and self.text_color is not None:
            classes = self.dataset_meta.get('classes', None)
            labels = instances.labels

            positions = bboxes[:, :2]
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'

                if isinstance(self.bbox_color,
                              tuple) and max(self.bbox_color) > 1:
                    facecolor = [c / 255.0 for c in self.bbox_color]
                else:
                    facecolor = self.bbox_color

                self.draw_texts(
                    label_text,
                    pos,
                    colors=self.text_color,
                    font_sizes=int(13 * scales[i]),
                    vertical_alignments='bottom',
                    bboxes=[{
                        'facecolor': facecolor,
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    def _draw_instances_kpts(self,
                             image: np.ndarray,
                             instances: InstanceData,
                             kpt_thr: float = 0.3,
                             show_kpt_idx: bool = False,
                             skeleton_style: str = 'mmpose',
                             side: int = 1,):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        # self.my_draw_instances_kpts(self, image,instances,kpt_thr,show_kpt_idx,skeleton_style)
        ## 定义点和线的颜色
        img_h, img_w, _ = image.shape
        full_skeleton = self.skeleton
        
        self.set_image(image)
        if instances.keypoints.shape[1] ==26:
            a = np.array([255,69,0], dtype=np.uint8)
            self.kpt_color = np.expand_dims(a,0).repeat(26,axis=0)
            b = np.array([51, 153, 255], dtype=np.uint8)
            self.link_color = np.expand_dims(b,0).repeat(27,axis=0)
            if skeleton_style == 'openpose':
                return self._draw_instances_kpts_openpose(image, instances,
                                                        kpt_thr)

            
            ###########################################
            print(self.side)
            if self.side == 1:
                #####正面
                whether_visible = np.ones((1,26))
                whether_visible[0,24] = 0
                whether_visible[0,25] = 0
                #####侧面左
            elif self.side == 2:
                whether_visible = np.array([[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
                #####侧面右
            elif self.side == 3:
                whether_visible = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
                #####背面
            elif self.side == 4:
                whether_visible = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]])
        
        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            for kpts, visible, whe in zip(keypoints, keypoints_visible, whether_visible):
                kpts = np.array(kpts, copy=False)
            ### 点改颜色
                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')
                #link 改颜色
                # draw links
                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')
                    if kpts.shape[0] ==26:
                        if self.side == 1:
                            self.skeleton = [(15, 13), (13, 11), (11, 19), (16, 14), (14, 12), (12, 19), (17, 18), (18, 19), (18, 5), (5, 7), (7, 9), (18, 6), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (15, 20), (15, 22), (16, 21), (16, 23)]
                        if self.side == 2:
                            self.skeleton = [(3,5), (5,11), (11,13), (13,15), (15,24), (15,20)]
                        if self.side == 3:
                            self.skeleton = [(4,6), (6,12), (12,14), (14,16), (16,25), (16,21)]
                        if self.side == 4:
                            self.skeleton = [(15, 13), (13, 11), (11, 19), (16, 14), (14, 12), (12, 19), (17, 18), (18, 19), (18, 5), (5, 7), (7, 9), (18, 6), (6, 8), (8, 10), (15,24), (16, 25)]
                    
                    for sk_id, sk in enumerate(self.skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h or visible[sk[0]] < kpt_thr
                                or visible[sk[1]] < kpt_thr
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue
                        #
                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            #######################################################改固定颜色
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0,
                                min(1,
                                    0.5 * (visible[sk[0]] + visible[sk[1]])))

                        self.draw_lines(
                            X, Y, color, line_widths=self.line_width)
                #点改颜色
                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if visible[kid] < kpt_thr or kpt_color[kid] is None or whe[kid] == 0:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    ###################################################改固定颜色
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, visible[kid]))
                    self.draw_circles(
                        kpt,
                        radius=np.array([self.radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=self.radius)
                    if show_kpt_idx:
                        kpt_idx_coords = kpt + [self.radius, -self.radius]
                        self.draw_texts(
                            str(kid),
                            kpt_idx_coords,
                            colors=color,
                            font_sizes=self.radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')
        img = self.get_image()
        if self.show_lines:
            if self.side ==1:
                points_horizon = kpts[[6,8,12,14,16],1]
                points_vertical = kpts[[19],0]
            elif self.side ==2:
                points_horizon = []
                points_vertical = kpts[[15],0]
            elif self.side ==3:
                points_horizon = []
                points_vertical = kpts[[16],0]
            elif self.side ==4:
                points_horizon = kpts[[5,7,11,13,15],1]
                points_vertical = kpts[[19],0]
            else:
                points_horizon = []
                points_vertical = []
            height, width = img.shape[:2]

            # 绘制水平线
            for y in points_horizon:
                y = int(y)
                cv2.line(img, (0, y), (width, y), color=(0,250,154), thickness=2)

            # 绘制垂直线
            for x in points_vertical:
                x = int(x)
                cv2.line(img, (x, 0), (x, height), color=(0,250,154), thickness=2)
        def cal_degree(i, j, d, thre, interval):
            x1, y1 = kpts[[i],0], kpts[[i],1]
            x2, y2 = kpts[[j],0], kpts[[j],1]

            # 计算连线的斜率
            delta_x = x2 - x1
            delta_y = y2 - y1

            # 计算连线与垂线之间的角度
            if d:
                angle_radians = math.atan2(delta_x, delta_y)  # 计算反正切，返回的是弧度
            else:
                angle_radians = math.atan2(delta_y, delta_x)  # 计算反正切，返回的是弧度
            angle_degrees = math.degrees(angle_radians)  # 转换为角度

            # 如果你只关心与水平线的夹角
            if angle_degrees > 90:
                angle_degrees = 180-angle_degrees
            if angle_degrees != 0:
                if self.side == 1 or self.side ==4:
                    label = '右'
                if self.side == 2:
                    label = '前'
                if self.side ==3:
                    label = '后'
            else:
                label = '-'
            if angle_degrees < 0.0:
                angle_degrees = min(abs(angle_degrees), 180.0 +angle_degrees)
                if self.side == 1 or self.side ==4:
                    label = '左'
                if self.side == 2:
                    label = '后'
                if self.side ==3:
                    label = '前'
            if angle_degrees < thre:
                score = '正常'
            else:
                if angle_degrees < thre + interval:
                    score = '轻微'
                else:
                    if angle_degrees < thre + interval*2:
                        score = '明显'
                    else:
                        score = '严重'
            return angle_degrees, label, score
        
        def calculate_angle_between_three_points(i, j, k, thre, interval):
            """
            计算三个点形成的夹角，并返回较小的角度。
            """
            xA, yA = kpts[[i],0], kpts[[i],1]
            xB, yB = kpts[[j],0], kpts[[j],1]
            xC, yC = kpts[[k],0], kpts[[k],1]
            # 向量 AB
            AB = (xA - xB, yA - yB)
            # 向量 BC
            BC = (xC - xB, yC - yB)

            # 计算向量的点积和模长
            dot_product = AB[0] * BC[0] + AB[1] * BC[1]
            magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
            magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)

            # 计算夹角的余弦值
            cosine_angle = dot_product / (magnitude_AB * magnitude_BC)

            # 反余弦得到角度，结果是弧度
            angle_radians = math.acos(cosine_angle)

            # 将弧度转换为度
            angle_degrees = math.degrees(angle_radians)

            # 使用叉积判断朝向
            cross_product = AB[0] * BC[1] - AB[1] * BC[0]
            
            if cross_product > 0:
                orientation = "逆时针"
            elif cross_product < 0:
                orientation = "顺时针"
            else:
                orientation = "共线"

            # 返回夹角和朝向
            # return angle_degrees, orientation

            # 确保返回的角度在 [0, 180) 范围内
            smaller_angle = min(angle_degrees, 180 - angle_degrees)
            angle_degrees = smaller_angle
            if angle_degrees < thre:
                score = '正常'
            else:
                if angle_degrees < thre + interval:
                    score = '轻微'
                else:
                    if angle_degrees < thre + interval*2:
                        score = '明显'
                    else:
                        score = '严重'
            return angle_degrees, score, orientation
        df = pd.DataFrame(columns=["id","degree","oren","level",'range', 'interval', 'class'])
        if self.side ==1:
               
            ## 头倾斜角度
            head_vertical, label, score = cal_degree(17, 18, 1, 4, 1.5)
            id = '头部倾斜'
            print(f"头部倾斜角度: {label}, {head_vertical}度   {score}")
            new_row = {"id": id,"degree": head_vertical,"oren": label,"level": score,'range': 1.3, 'interval':2.5, 'class':'头颈部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 头水平
            head_horizon, label, score = cal_degree(3, 4, 0, 4, 1.5)
            id = '头部水平'
            print(f"头水平倾斜角度: {label},  {head_horizon}度   {score}")
            new_row = {"id": id,"degree": head_horizon,"oren": label,"level": score,'range': 1.5, 'interval':2.4, 'class':'头颈部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 高低肩
            shoulder_horizon, label, score = cal_degree(5, 6, 0, 2.5, 1) 
            id = '双肩水平'
            print(f"高低肩倾斜角度: {label},  {shoulder_horizon}度   {score}")
            new_row = {"id": id,"degree": shoulder_horizon,"oren": label,"level": score,'range': 1.2, 'interval':2, 'class':'肩部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 脊柱角度
            spin_vertical, label, score = cal_degree(18, 19, 1, 3, 1.2)
            id = '躯干倾斜'
            print(f"躯干倾斜角度: {label}, {spin_vertical}度   {score}")
            new_row = {"id": id,"degree": spin_vertical,"oren": label,"level": score,'range': 1.2, 'interval':2, 'class':'躯干'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 骨盆
            till_horizon, label, score = cal_degree(11, 12, 0, 2, 1.5)
            id = '骨盆水平'
            print(f"骨盆倾斜角度: {label}, {till_horizon}度   {score}")
            new_row = {"id": id,"degree": till_horizon,"oren": label,"level": score,'range': 0.2, 'interval':1.6, 'class':'骨盆'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 膝盖水平
            knee_horizon, label, score = cal_degree(13, 14, 0, 2, 1.5)
            id = '膝关节水平'
            print(f"膝关节对线角度: {label}, {knee_horizon}度   {score}")
            new_row = {"id": id,"degree": knee_horizon,"oren": label,"level": score,'range': 0.2, 'interval':1.6, 'class':'腿部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 踝关节
            ankel_horizon, label, score = cal_degree(15, 16, 0, 2, 1.5)
            id = '踝关节水平'
            print(f"踝关节对线角度: {label}, {ankel_horizon}度   {score}")
            new_row = {"id": id,"degree": ankel_horizon,"oren": label,"level": score,'range': 0.2, 'interval':1.6, 'class':'脚部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            #################################################判断是否圆肩
            ## 肱骨角度
            left_arm_vertical, score, oren = calculate_angle_between_three_points(5, 7, 9, 6, 5)
            id = '左肱骨弯曲'
            print(f"左边肱骨角度: {left_arm_vertical}度   {score}")
            new_row = {"id": id,"degree": left_arm_vertical,"oren": oren,"level": score,'range': 3.5, 'interval':2, 'class':'肩部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 肱骨角度
            right_arm_vertical, score, oren = calculate_angle_between_three_points(6, 8, 10, 6, 5)
            id = '右肱骨弯曲'
            print(f"右边肱骨角度: {right_arm_vertical}度   {score}")
            new_row = {"id": id,"degree": right_arm_vertical,"oren": oren,"level": score,'range': 3.5, 'interval':2, 'class':'肩部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 左膝盖内外
            knee_vertical, score, oren = calculate_angle_between_three_points(11, 13, 15, 3, 3)
            id = '左腿Q角'
            if oren == '逆时针':
                oren = '内'
            else:
                oren = '外'
            print(f"左腿Q角度: {knee_vertical}度   {score} {oren}")
            new_row = {"id": id,"degree": knee_vertical+15,"oren": oren,"level": score,'range': 15, 'interval':5, 'class':'腿部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 右膝盖内外
            knee_vertical, score, oren = calculate_angle_between_three_points(12, 14, 16, 3, 3)
            id = '右腿Q角'
            if oren == '逆时针':
                oren = '外'
            else:
                oren = '内'
            print(f"右腿Q角度: {knee_vertical}度   {score} {oren}")
            new_row = {"id": id,"degree": knee_vertical+15,"oren": oren,"level": score,'range': 15, 'interval':5, 'class':'腿部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv('report/正面数据',encoding="utf_8_sig")
        if self.side ==2:
            ## 头前引
            neck_vertical, label, score = cal_degree(3, 5, 1, 6.5, 1.5)
            id = '颈椎倾斜'
            print(f"颈椎倾斜角度: {label}, {neck_vertical}度   {score}")
            new_row = {"id": id,"degree": neck_vertical,"oren": label,"level": score,'range': 1.9, 'interval':4.5, 'class':'头颈部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 肱骨前移
            shoulder_vertical, label, score = cal_degree(7, 5, 1, 5, 1.5)
            if label  == '前':
                label = '-'
            else:
                label = '前'
            id = '左肱骨位置'
            print(f"肱骨前移角度: {label}, {shoulder_vertical}度   {score}")
            new_row = {"id": id,"degree": shoulder_vertical,"oren": label,"level": score,'range': 3.5, 'interval':1, 'class':'肩部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            ## 膝超伸
            # if label  == '前':
            #     label = '后'
            # else:
            #     label = '前'
            id = '膝超伸'
            knee_vertical, label, score = cal_degree(13, 15, 1, 5, 1.5)
            print(f"膝超伸: {label}, {knee_vertical}度   {score}")
            new_row = {"id": id,"degree": knee_vertical,"oren": label,"level": score,'range': 3.5, 'interval':1.5, 'class':'腿部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            ## 重心前移
            id = '左重心前移'
            gravity_front, label, score = cal_degree(5, 15, 1, 5, 1.5)
            print(f"左重心前移: {label}, {gravity_front}度   {score}")
            new_row = {"id": id,"degree": gravity_front,"oren": label,"level": score,'range': 3.5, 'interval':1.5, 'class':'躯干'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            ## 骨盆前移
            id = '骨盆前/后移'
            till_front, label, score = cal_degree(11, 15, 1, 3.5, 1)

            print(f"骨盆前移角度: {label}, {till_front}度   {score}")
            new_row = {"id": id,"degree": till_front,"oren": label,"level": score,'range': 2.5, 'interval':1, 'class':'骨盆'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 骨盆前倾
            pelvis_tilt, score, oren = calculate_angle_between_three_points(5, 11, 15, 6, 2)
            id = '骨盆前/后倾'
            if oren == '逆时针':
                oren = '前'
            else:
                oren = '后'

            print(f"骨盆{oren}倾角度: {oren} {pelvis_tilt}度   {score}")
            new_row = {"id": id,"degree": pelvis_tilt+11,"oren": oren,"level": score,'range': 12, 'interval':5, 'class':'骨盆'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv('report/左侧面数据',encoding="utf_8_sig")
        if self.side ==3:
            ## 头前引
            neck_vertical, label, score = cal_degree(4, 6, 1, 6.5, 1.5)
            id = '颈椎倾斜'
            print(f"颈椎倾斜角度: {label}, {neck_vertical}度   {score}")
            new_row = {"id": id,"degree": neck_vertical,"oren": label,"level": score,'range': 1.9, 'interval':4.5, 'class':'头颈部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 肱骨前移
            shoulder_vertical, label, score = cal_degree(8, 6, 1, 5, 1.5)
            if label  == '前':
                label = '后'
            else:
                label = '前'
            id = '右肱骨位置'
            print(f"肱骨前移角度: {label}, {shoulder_vertical}度   {score}")
            new_row = {"id": id,"degree": shoulder_vertical,"oren": label,"level": score,'range': 3.5, 'interval':1, 'class':'肩部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            ## 膝超伸
            knee_vertical, label, score = cal_degree(14, 16, 1, 5, 1.5)
            # if label  == '前':
            #     label = '后'
            # else:
            #     label = '前'
            id = '膝超伸'
            print(f"膝超伸角度: {label}, {knee_vertical}度   {score}")
            new_row = {"id": id,"degree": knee_vertical,"oren": label,"level": score,'range': 3.5, 'interval':1.5, 'class':'腿部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            ## 重心前移
            id = '右重心前移'
            gravity_front, label, score = cal_degree(6, 16, 1, 5, 1.5)
            print(f"右重心前移: {label}, {gravity_front}度   {score}")
            new_row = {"id": id,"degree": gravity_front,"oren": label,"level": score,'range': 3.5, 'interval':1.5, 'class':'躯干'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            ## 骨盆前移
            id = '骨盆前/后移'
            till_front, label, score = cal_degree(12, 16, 1, 3.5, 1.5)
            print(f"骨盆前移角度: {label}, {till_front}度   {score}")
            new_row = {"id": id,"degree": till_front,"oren": label,"level": score,'range': 2.5, 'interval':1, 'class':'骨盆'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 骨盆前倾
            pelvis_tilt, score, oren = calculate_angle_between_three_points(6, 12, 16, 6, 2)
            id = '骨盆前/后倾'
            if oren == '逆时针':
                oren = '后'
            else:
                oren = '前'
            print(f"骨盆{oren}倾角度: {pelvis_tilt}度   {score}")
            new_row = {"id": id,"degree": pelvis_tilt+11,"oren": label,"level": score,'range': 12, 'interval':5, 'class':'骨盆'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv('report/右侧面数据',encoding="utf_8_sig")
        if self.side ==4:
            ## 骨盆
            till_horizon, label, score = cal_degree(11, 12, 0, 2, 1.5)
            id = '骨盆水平'
            print(f"骨盆倾斜角度: {label}, {till_horizon}度   {score}")
            new_row = {"id": id,"degree": till_horizon,"oren": label,"level": score,'range': 0.2, 'interval':1.6, 'class':'骨盆'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 左足弓
            foot_vertical, label, score = cal_degree(15, 24, 1, 5, 1)
            id = '左足内/外翻'
            oren = ''
            if label =='左':
                oren = '外'
            else:
                oren = '内'
            print(f"足{oren}倾斜角度: {label}, {foot_vertical}度   {score}")
            new_row = {"id": id,"degree": foot_vertical,"oren": oren,"level": score,'range': 5, 'interval':4, 'class':'脚部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            ## 右足弓
            foot_vertical, label, score = cal_degree(16, 25, 1, 5, 1)
            id = '右足内/外翻'
            if label =='左':
                oren = '内'
            else:
                oren = '外'
            print(f"足{oren}倾斜角度: {label}, {foot_vertical}度   {score}")
            new_row = {"id": id,"degree": foot_vertical,"oren": oren,"level": score,'range': 5, 'interval':4, 'class':'脚部'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv('report/背面数据',encoding="utf_8_sig")


        ##############################################
       
        self.skeleton = full_skeleton
        transparent_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        self.set_image(transparent_image)
        if instances.keypoints.shape[1] ==26:
            a = np.array([255,69,0], dtype=np.uint8)
            self.kpt_color = np.expand_dims(a,0).repeat(26,axis=0)
            b = np.array([51, 153, 255], dtype=np.uint8)
            self.link_color = np.expand_dims(b,0).repeat(27,axis=0)
            if skeleton_style == 'openpose':
                return self._draw_instances_kpts_openpose(image, instances,
                                                        kpt_thr)

            
            ###########################################
            # print(self.side)
            if self.side == 1:
                #####正面
                whether_visible = np.ones((1,26))
                whether_visible[0,24] = 0
                whether_visible[0,25] = 0
           
                if 'keypoints' in instances:
                    keypoints = instances.get('transformed_keypoints',
                                            instances.keypoints)

                    if 'keypoints_visible' in instances:
                        keypoints_visible = instances.keypoints_visible
                    else:
                        keypoints_visible = np.ones(keypoints.shape[:-1])

                    for kpts, visible, whe in zip(keypoints, keypoints_visible, whether_visible):
                        kpts = np.array(kpts, copy=False)
                    ### 点改颜色
                        if self.kpt_color is None or isinstance(self.kpt_color, str):
                            kpt_color = [self.kpt_color] * len(kpts)
                        elif len(self.kpt_color) == len(kpts):
                            kpt_color = self.kpt_color
                        else:
                            raise ValueError(
                                f'the length of kpt_color '
                                f'({len(self.kpt_color)}) does not matches '
                                f'that of keypoints ({len(kpts)})')
                        #link 改颜色
                        # draw links
                        if self.skeleton is not None and self.link_color is not None:
                            if self.link_color is None or isinstance(
                                    self.link_color, str):
                                link_color = [self.link_color] * len(self.skeleton)
                            elif len(self.link_color) == len(self.skeleton):
                                link_color = self.link_color
                            else:
                                raise ValueError(
                                    f'the length of link_color '
                                    f'({len(self.link_color)}) does not matches '
                                    f'that of skeleton ({len(self.skeleton)})')
                            if kpts.shape[0] ==26:
                                if self.side == 1:
                                    self.skeleton = [(15, 13), (13, 11), (11, 19), (16, 14), (14, 12), (12, 19), (17, 18), (18, 19), (18, 5), (5, 7), (7, 9), (18, 6), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (15, 20), (15, 22), (16, 21), (16, 23)]
                                if self.side == 2:
                                    self.skeleton = [(3,5), (5,11), (11,13), (13,15), (15,24), (15,20)]
                                if self.side == 3:
                                    self.skeleton = [(4,6), (6,12), (12,14), (14,16), (16,25), (16,21)]
                                if self.side == 4:
                                    self.skeleton = [(15, 13), (13, 11), (11, 19), (16, 14), (14, 12), (12, 19), (17, 18), (18, 19), (18, 5), (5, 7), (7, 9), (18, 6), (6, 8), (8, 10), (15,24), (16, 25)]
                            
                            for sk_id, sk in enumerate(self.skeleton):
                                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                        or pos1[1] >= img_h or pos2[0] <= 0
                                        or pos2[0] >= img_w or pos2[1] <= 0
                                        or pos2[1] >= img_h or visible[sk[0]] < kpt_thr
                                        or visible[sk[1]] < kpt_thr
                                        or link_color[sk_id] is None):
                                    # skip the link that should not be drawn
                                    continue
                                #
                                X = np.array((pos1[0], pos2[0]))
                                Y = np.array((pos1[1], pos2[1]))
                                color = link_color[sk_id]
                                if not isinstance(color, str):
                                    #######################################################改固定颜色
                                    color = tuple(int(c) for c in color)
                                transparency = self.alpha
                                if self.show_keypoint_weight:
                                    transparency *= max(
                                        0,
                                        min(1,
                                            0.5 * (visible[sk[0]] + visible[sk[1]])))

                                self.draw_lines(
                                    X, Y, color, line_widths=self.line_width)
                        #点改颜色
                        # draw each point on image
                        for kid, kpt in enumerate(kpts):
                            if visible[kid] < kpt_thr or kpt_color[kid] is None or whe[kid] == 0:
                                # skip the point that should not be drawn
                                continue

                            color = kpt_color[kid]
                            ###################################################改固定颜色
                            if not isinstance(color, str):
                                color = tuple(int(c) for c in color)
                            transparency = self.alpha
                            if self.show_keypoint_weight:
                                transparency *= max(0, min(1, visible[kid]))
                            self.draw_circles(
                                kpt,
                                radius=np.array([self.radius]),
                                face_colors=color,
                                edge_colors=color,
                                alpha=transparency,
                                line_widths=self.radius)
                            if show_kpt_idx:
                                kpt_idx_coords = kpt + [self.radius, -self.radius]
                                self.draw_texts(
                                    str(kid),
                                    kpt_idx_coords,
                                    colors=color,
                                    font_sizes=self.radius * 3,
                                    vertical_alignments='bottom',
                                    horizontal_alignments='center')
        full_kpts = np.squeeze(instances.keypoints)
        skeleton_img = self.get_image()
        
        # skeleton_img, af_kpts = process_image_and_keypoints(skeleton_img, full_kpts)
        np.save('demo/resources/full_kpts.npy', full_kpts)
        cv2.imwrite('demo/resources/skeleton_org.png',skeleton_img)
        return img
    

############################修改
    def _draw_instances_kpts_openpose(self,
                                      image: np.ndarray,
                                      instances: InstanceData,
                                      kpt_thr: float = 0.3):
        """Draw keypoints and skeletons (optional) of GT or prediction in
        openpose style.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        self.set_image(image)
        img_h, img_w, _ = image.shape

        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            keypoints_info = np.concatenate(
                (keypoints, keypoints_visible[..., None]), axis=-1)
            # compute neck joint
            neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
            # neck score when visualizing pred
            neck[:, 2:3] = np.logical_and(
                keypoints_info[:, 5, 2:3] > kpt_thr,
                keypoints_info[:, 6, 2:3] > kpt_thr).astype(int)
            new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

            mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
            openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            new_keypoints_info[:, openpose_idx] = \
                new_keypoints_info[:, mmpose_idx]
            keypoints_info = new_keypoints_info

            keypoints, keypoints_visible = keypoints_info[
                ..., :2], keypoints_info[..., 2]

            for kpts, visible in zip(keypoints, keypoints_visible):
                kpts = np.array(kpts, copy=False)

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                # draw links
                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h or visible[sk[0]] < kpt_thr
                                or visible[sk[1]] < kpt_thr
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue

                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0,
                                min(1,
                                    0.5 * (visible[sk[0]] + visible[sk[1]])))

                        if sk_id <= 16:
                            # body part
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            transparency = 0.6
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            polygons = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(self.line_width)),
                                int(angle), 0, 360, 1)

                            self.draw_polygons(
                                polygons,
                                edge_colors=color,
                                face_colors=color,
                                alpha=transparency)

                        else:
                            # hand part
                            self.draw_lines(X, Y, color, line_widths=2)

                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if visible[kid] < kpt_thr or kpt_color[
                            kid] is None or kpt_color[kid].sum() == 0:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, visible[kid]))

                    # draw smaller dots for face & hand keypoints
                    radius = self.radius // 2 if kid > 17 else self.radius

                    self.draw_circles(
                        kpt,
                        radius=np.array([radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=radius)

        return self.get_image()

    def _draw_instance_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[np.ndarray] = None,
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
                pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        if 'heatmaps' not in fields:
            return None
        heatmaps = fields.heatmaps
        if isinstance(heatmaps, np.ndarray):
            heatmaps = torch.from_numpy(heatmaps)
        if heatmaps.dim() == 3:
            heatmaps, _ = heatmaps.max(dim=0)
        heatmaps = heatmaps.unsqueeze(0)
        out_image = self.draw_featmap(heatmaps, overlaid_image)
        return out_image

    def _draw_instance_xy_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[np.ndarray] = None,
        n: int = 20,
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
            pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.
            n (int): Number of keypoint, up to 20.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        if 'heatmaps' not in fields:
            return None
        heatmaps = fields.heatmaps
        _, h, w = heatmaps.shape
        if isinstance(heatmaps, np.ndarray):
            heatmaps = torch.from_numpy(heatmaps)
        out_image = SimCCVisualizer().draw_instance_xy_heatmap(
            heatmaps, overlaid_image, n)
        out_image = cv2.resize(out_image[:, :, ::-1], (w, h))
        return out_image

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: PoseDataSample,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_heatmap: bool = False,
                       draw_bbox: bool = False,
                       show_kpt_idx: bool = False,
                       skeleton_style: str = 'mmpose',
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       kpt_thr: float = 0.3,
                       side: int = 1,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier
            image (np.ndarray): The image to draw
            data_sample (:obj:`PoseDataSample`, optional): The data sample
                to visualize
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            draw_pred (bool): Whether to draw Prediction PoseDataSample.
                Defaults to ``True``
            draw_bbox (bool): Whether to draw bounding boxes. Default to
                ``False``
            draw_heatmap (bool): Whether to draw heatmaps. Defaults to
                ``False``
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``
            show (bool): Whether to display the drawn image. Default to
                ``False``
            wait_time (float): The interval of show (s). Defaults to 0
            out_file (str): Path to output file. Defaults to ``None``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            step (int): Global step value to record. Defaults to 0
        """

        gt_img_data = None
        pred_img_data = None

        if draw_gt:
            gt_img_data = image.copy()
            gt_img_heatmap = None

            # draw bboxes & keypoints
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances_kpts(
                    gt_img_data, data_sample.gt_instances, kpt_thr,
                    show_kpt_idx, skeleton_style, side)
                if draw_bbox:
                    gt_img_data = self._draw_instances_bbox(
                        gt_img_data, data_sample.gt_instances)

            # draw heatmaps
            if 'gt_fields' in data_sample and draw_heatmap:
                gt_img_heatmap = self._draw_instance_heatmap(
                    data_sample.gt_fields, image)
                if gt_img_heatmap is not None:
                    gt_img_data = np.concatenate((gt_img_data, gt_img_heatmap),
                                                 axis=0)

        if draw_pred:
            pred_img_data = image.copy()
            pred_img_heatmap = None

            # draw bboxes & keypoints
            if 'pred_instances' in data_sample:
                pred_img_data = self._draw_instances_kpts(
                    pred_img_data, data_sample.pred_instances, kpt_thr,
                    show_kpt_idx, skeleton_style)
                if draw_bbox:
                    pred_img_data = self._draw_instances_bbox(
                        pred_img_data, data_sample.pred_instances)

            # draw heatmaps
            if 'pred_fields' in data_sample and draw_heatmap:
                if 'keypoint_x_labels' in data_sample.pred_instances:
                    pred_img_heatmap = self._draw_instance_xy_heatmap(
                        data_sample.pred_fields, image)
                else:
                    pred_img_heatmap = self._draw_instance_heatmap(
                        data_sample.pred_fields, image)
                if pred_img_heatmap is not None:
                    pred_img_data = np.concatenate(
                        (pred_img_data, pred_img_heatmap), axis=0)

        # merge visualization results
        if gt_img_data is not None and pred_img_data is not None:
            if gt_img_heatmap is None and pred_img_heatmap is not None:
                gt_img_data = np.concatenate((gt_img_data, image), axis=0)
            elif gt_img_heatmap is not None and pred_img_heatmap is None:
                pred_img_data = np.concatenate((pred_img_data, image), axis=0)

            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)

        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            # save drawn_img to backends
            self.add_image(name, drawn_img, step)

        return self.get_image()
