# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Long-memory sampler matching the reviewed SurgicalSAM3 strategy."""

import random

from training.dataset.vos_sampler import (
    MAX_RETRIES,
    SampledFramesAndObjects,
    VOSSampler,
    get_category_ids,
    get_visible_object_ids,
)


class RandomUniformSampler(VOSSampler):
    """Sample recent and long-memory frames as in SurgicalSAM2.1/SAM3."""

    def __init__(
        self,
        num_short_frames,
        num_long_frames,
        max_num_objects,
        reverse_time_prob=0.0,
        step_length=1,
        use_tissue_text=False,
        allow_replacement_for_short_videos=True,
    ):
        if num_short_frames < 1:
            raise ValueError("num_short_frames must be at least 1")
        if num_long_frames < 1:
            raise ValueError("num_long_frames must be at least 1")
        if step_length < 1:
            raise ValueError("step_length must be at least 1")

        self.num_short_frames = num_short_frames
        self.num_long_frames = num_long_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob
        self.step_length = step_length
        self.use_tissue_text = use_tissue_text
        self.allow_replacement_for_short_videos = (
            allow_replacement_for_short_videos
        )

    @property
    def num_frames(self):
        return self.num_short_frames + self.num_long_frames

    def sample(self, video, segment_loader, epoch=None):
        del epoch
        visible_object_ids = []
        frames = None

        if not video.frames:
            raise ValueError(f"Video {video.video_name} has no frames")

        for _ in range(MAX_RETRIES):
            try:
                if len(video.frames) < self.num_frames:
                    # This is the original short-video behavior: sample from
                    # the complete video with replacement and sort by time.
                    if not self.allow_replacement_for_short_videos:
                        continue
                    indices = random.choices(
                        range(len(video.frames)), k=self.num_frames
                    )
                    indices.sort()
                    frames = [video.frames[i] for i in indices]
                else:
                    # First sample the short/recent block.
                    start = random.randrange(
                        self.num_long_frames,
                        len(video.frames) - self.num_short_frames + 1,
                    )
                    this_frame_idx = start
                    short_mem_indices = [start]
                    for _ in range(self.num_short_frames - 1):
                        this_frame_idx += random.randint(1, self.step_length)
                        short_mem_indices.append(this_frame_idx)

                    # Retry if the sampled short block crosses the boundary.
                    if short_mem_indices[-1] >= len(video.frames):
                        continue

                    if random.random() < self.reverse_time_prob:
                        # Sample long-memory frames after the short block, then
                        # reverse the complete stream for reverse-time training.
                        if (
                            short_mem_indices[-1] + self.num_long_frames
                            >= len(video.frames)
                        ):
                            continue
                        long_mem_indices = random.sample(
                            range(short_mem_indices[-1] + 1, len(video.frames)),
                            k=self.num_long_frames,
                        )
                        long_mem_indices.sort()
                        frame_indices = (
                            short_mem_indices + long_mem_indices
                        )[::-1]
                    else:
                        # Sample long-memory frames before the short block.
                        long_mem_indices = random.sample(
                            range(0, start), k=self.num_long_frames
                        )
                        long_mem_indices.sort()
                        frame_indices = long_mem_indices + short_mem_indices

                    frames = [video.frames[i] for i in frame_indices]

                # The shared helper checks the first frame for visible objects.
                visible_object_ids = get_visible_object_ids(
                    segment_loader, frames[0]
                )
                if visible_object_ids:
                    break
            except Exception:
                # Match the reviewed reference behavior: retry a different
                # temporal draw when sampling or annotation lookup fails.
                continue

        if not visible_object_ids or frames is None:
            raise RuntimeError(
                f"No visible objects found for a long-range clip from "
                f"{video.video_name} after {MAX_RETRIES} retries"
            )

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
            is_long_mem_batch=True,
            # The first long-memory frame is the conditional frame in the
            # current SurgSLOT model, so only the remaining frames are memory.
            num_long_mem_frames=self.num_long_frames - 1,
        )


class LongRangeSampler(RandomUniformSampler):
    """Backward-compatible name used by older SurgSLOT configs."""

