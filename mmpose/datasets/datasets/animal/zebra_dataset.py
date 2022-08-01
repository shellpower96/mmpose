# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

import numpy as np

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class ZebraDataset(BaseCocoStyleDataset):
    """ZebraDataset for animal pose estimation.

    "DeepPoseKit, a software toolkit for fast and robust animal
    pose estimation using deep learning" Elife'2019.
    More details can be found in the `paper
    <https://elifesciences.org/articles/47994>`__ .

    Zebra keypoints::

        0: "snout",
        1: "head",
        2: "neck",
        3: "forelegL1",
        4: "forelegR1",
        5: "hindlegL1",
        6: "hindlegR1",
        7: "tailbase",
        8: "tailtip"

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/zebra.py')

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw Zebra annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        img_path = osp.join(self.data_prefix['img_path'], img['file_name'])
        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        # use the entire image which is 160x160
        bbox = np.array([0, 0, 160, 160], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        num_keypoints = ann['num_keypoints']

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img_path,
            'img_shape': (img_h, img_w, 3),
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann['iscrowd'],
            'id': ann['id'],
        }

        return data_info