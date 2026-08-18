# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List, Optional

from training.dataset.vos_segment_loader import LazySegments
from training.utils.category_name_list import get_category_id

MAX_RETRIES = 1000


def get_visible_object_ids(segment_loader, frame):
    """Return object ids with a non-empty mask on ``frame``."""
    loaded_segments = segment_loader.load(frame.frame_idx)
    if isinstance(loaded_segments, LazySegments):
        return list(loaded_segments.keys())
    return [
        object_id
        for object_id, segment in loaded_segments.items()
        if segment.sum()
    ]


def get_category_ids(video, object_ids, use_tissue_text=False):
    """Map sampled object ids to the unified SurgSLOT category vocabulary.

    Datasets without category metadata receive ``-1`` and are ignored by the
    vision-language loss while still contributing segmentation supervision.
    """
    metadata = getattr(video, "meta_annotation", None)
    categories = None
    if metadata is not None:
        categories = metadata.get("info", {}).get("category")
    if not categories:
        return [-1] * len(object_ids)
    return [
        get_category_id(
            categories.get(str(object_id), categories.get(object_id, "unknown")),
            use_tissue_text=use_tissue_text,
        )
        for object_id in object_ids
    ]


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]
    category_ids: Optional[List[int]] = None
    is_long_mem_batch: bool = False
    num_long_mem_frames: int = 0


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class RandomUniformSampler(VOSSampler):
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
        step_length=1,
        use_tissue_text=False,
    ):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob
        if step_length < 1:
            raise ValueError("step_length must be at least 1")
        self.step_length = step_length
        self.use_tissue_text = use_tissue_text

    def sample(self, video, segment_loader, epoch=None):

        for retry in range(MAX_RETRIES):
            if not video.frames:
                raise ValueError(f"Video {video.video_name} has no frames")
            if len(video.frames) < self.num_frames:
                # Static/very short surgical clips remain trainable by sampling
                # with replacement, matching the mixed image-video setup.
                indices = sorted(
                    random.choices(range(len(video.frames)), k=self.num_frames)
                )
            else:
                start = random.randrange(0, len(video.frames) - self.num_frames + 1)
                indices = [start]
                for _ in range(self.num_frames - 1):
                    indices.append(indices[-1] + random.randint(1, self.step_length))
                if indices[-1] >= len(video.frames):
                    continue
            frames = [video.frames[index] for index in indices]
            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]

            # Get first frame object ids
            visible_object_ids = get_visible_object_ids(segment_loader, frames[0])

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(
            frames=frames,
            object_ids=object_ids,
            category_ids=get_category_ids(
                video, object_ids, use_tissue_text=self.use_tissue_text
            ),
        )


class EvalSampler(VOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames
        object_ids = segment_loader.load(frames[0].frame_idx).keys()
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
