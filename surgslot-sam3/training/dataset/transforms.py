# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms and data augmentation for both image + bbox.
"""

import logging

import random
from typing import Iterable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage

from torchvision.transforms import InterpolationMode

from training.utils.data_utils import VideoDatapoint

import math
from typing import Iterable, Tuple, List, Union
from torchvision.transforms import InterpolationMode


def hflip(datapoint, index):

    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)

    return datapoint


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)


def resize(datapoint, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        cur_size = (
            datapoint.frames[index].data.size()[-2:][::-1]
            if v2
            else datapoint.frames[index].data.size
        )
        size = get_size(cur_size, size, max_size)

    old_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )
    if v2:
        datapoint.frames[index].data = Fv2.resize(
            datapoint.frames[index].data, size, antialias=True
        )
    else:
        datapoint.frames[index].data = F.resize(datapoint.frames[index].data, size)

    new_size = (
        datapoint.frames[index].data.size()[-2:][::-1]
        if v2
        else datapoint.frames[index].data.size
    )

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.resize(obj.segment[None, None], size).squeeze()

    h, w = size
    datapoint.frames[index].size = (h, w)
    return datapoint


def pad(datapoint, index, padding, v2=False):
    old_h, old_w = datapoint.frames[index].size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data, (0, 0, padding[0], padding[1])
        )
        h += padding[1]
        w += padding[0]
    else:
        # left, top, right, bottom
        datapoint.frames[index].data = F.pad(
            datapoint.frames[index].data,
            (padding[0], padding[1], padding[2], padding[3]),
        )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    datapoint.frames[index].size = (h, w)

    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = Fv2.pad(obj.segment, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))
    return datapoint


class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = hflip(datapoint, i)
            return datapoint
        for i in range(len(datapoint.frames)):
            if random.random() < self.p:
                datapoint = hflip(datapoint, i)
        return datapoint


class RandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(datapoint.frames)):
                datapoint = resize(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.frames)):
            size = random.choice(self.sizes)
            datapoint = resize(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint


class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)
            else:
                img.data = F.to_tensor(img.data)
        return datapoint


class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for img in datapoint.frames:
            if self.v2:
                img.data = Fv2.convert_image_dtype(img.data, torch.float32)
                img.data = Fv2.normalize(img.data, mean=self.mean, std=self.std)
            else:
                img.data = F.normalize(img.data, mean=self.mean, std=self.std)

        return datapoint


class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datapoint, **kwargs):
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T.Grayscale(num_output_channels=3)

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for img in datapoint.frames:
                    img.data = self.Grayscale(img.data)
            return datapoint
        for img in datapoint.frames:
            if random.random() < self.p:
                img.data = self.Grayscale(img.data)
        return datapoint


class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else ([-hue, hue])

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        if self.consistent_transform:
            # Create a color jitter transformation params
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for img in datapoint.frames:
            if not self.consistent_transform:
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img.data = F.adjust_brightness(img.data, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img.data = F.adjust_contrast(img.data, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img.data = F.adjust_saturation(img.data, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img.data = F.adjust_hue(img.data, hue_factor)
        return datapoint


class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        """
        The mask is required for this transform.
        if consistent_transform if True, then the same random affine is applied to all frames and masks.
        """
        self.degrees = degrees if isinstance(degrees, list) else ([-degrees, degrees])
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self.fill_img = image_mean
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, datapoint: VideoDatapoint, **kwargs):
        for _tentative in range(self.num_tentatives):
            res = self.transform_datapoint(datapoint)
            if res is not None:
                return res

        if self.log_warning:
            logging.warning(
                f"Skip RandomAffine for zero-area mask in first frame after {self.num_tentatives} tentatives"
            )
        return datapoint

    def transform_datapoint(self, datapoint: VideoDatapoint):
        _, height, width = F.get_dimensions(datapoint.frames[0].data)
        img_size = [width, height]

        if self.consistent_transform:
            # Create a random affine transformation
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(datapoint.frames):
            this_masks = [
                obj.segment.unsqueeze(0) if obj.segment is not None else None
                for obj in img.objects
            ]
            if not self.consistent_transform:
                # if not consistent we create a new affine params for every frame&mask pair Create a random affine transformation
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )

            transformed_bboxes, transformed_masks = [], []
            for i in range(len(img.objects)):
                if this_masks[i] is None:
                    transformed_masks.append(None)
                    # Dummy bbox for a dummy target
                    transformed_bboxes.append(torch.tensor([[0, 0, 1, 1]]))
                else:
                    transformed_mask = F.affine(
                        this_masks[i],
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0.0,
                    )
                    if img_idx == 0 and transformed_mask.max() == 0:
                        # We are dealing with a video and the object is not visible in the first frame
                        # Return the datapoint without transformation
                        return None
                    transformed_masks.append(transformed_mask.squeeze())

            for i in range(len(img.objects)):
                img.objects[i].segment = transformed_masks[i]

            img.data = F.affine(
                img.data,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )
        return datapoint


def random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    # Step 1: downsize the images and paste them into a mosaic
    image_data = datapoint.frames[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,  # antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.frames[index].data = image_data_output

    # Step 2: downsize the masks and paste them into the target grid of the mosaic
    for obj in datapoint.frames[index].objects:
        if obj.segment is None:
            continue
        assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
        segment_output = torch.zeros_like(obj.segment)

        target_y_offset_b = target_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.frames)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint



SizeArg = Union[int, Tuple[int, int]]


def _to_hw(size: SizeArg) -> Tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    assert isinstance(size, (tuple, list)) and len(size) == 2
    return (size[0], size[1])  # (h, w)


def _img_hw(img, v2: bool) -> Tuple[int, int]:
    # return (h, w)
    if v2:
        return img.shape[-2], img.shape[-1]
    else:
        # PIL.Image.size -> (w, h)
        w, h = img.size
        return h, w


def _sample_crop_params(h: int, w: int,
                        scale: Tuple[float, float],
                        ratio: Tuple[float, float]) -> Tuple[int, int, int, int]:
    """Return (top, left, ch, cw). Closely follows torchvision logic with fallbacks."""
    area = h * w
    log_ratio_min, log_ratio_max = math.log(ratio[0]), math.log(ratio[1])

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = math.exp(random.uniform(log_ratio_min, log_ratio_max))
        cw = int(round(math.sqrt(target_area * aspect)))
        ch = int(round(math.sqrt(target_area / aspect)))
        if 0 < cw <= w and 0 < ch <= h:
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            return top, left, ch, cw

    # Fallback: center-crop to the closest acceptable aspect within bounds
    in_ratio = w / h
    rmin, rmax = ratio
    if in_ratio < rmin:
        cw = w
        ch = int(round(cw / rmin))
        if ch > h:
            ch = h
            cw = int(round(ch * rmin))
    elif in_ratio > rmax:
        ch = h
        cw = int(round(ch * rmax))
        if cw > w:
            cw = w
            ch = int(round(cw / rmax))
    else:
        ch, cw = h, w

    top = (h - ch) // 2
    left = (w - cw) // 2
    return top, left, ch, cw


def _count_objects_after_crop_no_mutation(frame, crop_params, min_object_pixels: int = 1):
    """Return object indices that retain foreground pixels after a trial crop."""
    top, left, ch, cw = crop_params
    kept = []
    for idx, obj in enumerate(getattr(frame, "objects", [])):
        seg = getattr(obj, "segment", None)
        # Inspect the crop window at the original resolution without mutation.
        sub = seg[top:top+ch, left:left+cw]
        if int(sub.sum().item()) >= min_object_pixels:
            kept.append(idx)
    return kept

class RandomResizedCropAPI:
    def __init__(
        self,
        sizes: Union[SizeArg, Iterable[SizeArg]],
        consistent_transform: bool,
        scale: Tuple[float, float] = (0.49, 1.0),
        ratio: Tuple[float, float] = (1, 16/9),  #
        v2: bool = False,
        max_tries: int = 30,
        min_object_pixels: int = 1,
        remove_empty: bool = True,
    ):
        if isinstance(sizes, (int, tuple, list)):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes: List[Tuple[int, int]] = [_to_hw(s) for s in sizes]
        self.consistent_transform = consistent_transform
        self.scale = scale
        self.ratio = ratio
        self.v2 = v2
        self.max_tries = max_tries
        self.min_object_pixels = min_object_pixels
        self.remove_empty = remove_empty

    def __call__(self, datapoint, **kwargs):
        n = len(datapoint.frames)

        if self.consistent_transform:
            out_h, out_w = random.choice(self.sizes)
            base_h, base_w = _img_hw(datapoint.frames[0].data, self.v2)

            # Sample and validate a crop on frame 0 without modifying the data.
            chosen_params = None
            kept_indices = None
            for _ in range(self.max_tries):
                params = _sample_crop_params(base_h, base_w, self.scale, self.ratio)
                kept = _count_objects_after_crop_no_mutation(
                    datapoint.frames[0], params, self.min_object_pixels
                )
                if len(kept) > 0:
                    chosen_params = params
                    kept_indices = set(kept)
                    break

            # Fall back to the full image, then resize to the requested output size.
            if chosen_params is None:
                logging.warning("Skip RandomResizedCrop: no surviving object in frame-0.")
                chosen_params = (0, 0, base_h, base_w)
                # The full-image fallback retains every object.
                kept_indices = set(range(len(getattr(datapoint.frames[0], "objects", []))))

            # Remove category labels for objects excluded by the crop.
            if datapoint.category_ids is not None:
                datapoint.category_ids = [cat_id for j, cat_id in enumerate(datapoint.category_ids) if j in kept_indices]
            # Apply the selected crop consistently to every frame.
            for i in range(n):
                h, w = _img_hw(datapoint.frames[i].data, self.v2)
                # Resample locally when frame resolution differs from frame 0.
                params = chosen_params if (h, w) == (base_h, base_w) else \
                         _sample_crop_params(h, w, self.scale, self.ratio)
                # Remove objects that did not survive the reference crop.
                datapoint.frames[i].objects = [
                    obj for j, obj in enumerate(datapoint.frames[i].objects) if j in kept_indices
                ]
                datapoint = self._apply_once(datapoint, i, params, (out_h, out_w))

            return datapoint

        # In non-consistent mode, sample each frame independently.
        for i in range(n):
            out_h, out_w = random.choice(self.sizes)
            h, w = _img_hw(datapoint.frames[i].data, self.v2)
            params = _sample_crop_params(h, w, self.scale, self.ratio)
            datapoint = self._apply_once(datapoint, i, params, (out_h, out_w))
            if self.remove_empty:
                # Optionally remove empty objects on each frame.
                new_objs = []
                for obj in getattr(datapoint.frames[i], "objects", []):
                    seg = getattr(obj, "segment", None)
                    if seg is None:
                        continue
                    if seg.sum().item() > 0:
                        new_objs.append(obj)
                datapoint.frames[i].objects = new_objs
        return datapoint

    def _apply_once(self, datapoint, index: int,
                    crop_params: Tuple[int, int, int, int],
                    out_size: Tuple[int, int]):
        top, left, ch, cw = crop_params
        out_h, out_w = out_size

        # --- image ---
        if self.v2:
            img = Fv2.crop(datapoint.frames[index].data, top, left, ch, cw)
            img = Fv2.resize(img, (out_h, out_w), antialias=True)
        else:
            img = F.crop(datapoint.frames[index].data, top, left, ch, cw)
            img = F.resize(img, (out_h, out_w))
        datapoint.frames[index].data = img

        # --- masks ---
        for obj in getattr(datapoint.frames[index], "objects", []):
            if getattr(obj, "segment", None) is not None:
                if self.v2:
                    seg = Fv2.crop(obj.segment[None, None], top, left, ch, cw).squeeze()
                    seg = Fv2.resize(
                        seg[None, None], (out_h, out_w),
                        interpolation=InterpolationMode.NEAREST, antialias=False
                    ).squeeze()
                else:
                    seg = F.crop(obj.segment[None, None], top, left, ch, cw).squeeze()
                    seg = F.resize(
                        seg[None, None], (out_h, out_w),
                        interpolation=InterpolationMode.NEAREST
                    ).squeeze()
                obj.segment = seg

        datapoint.frames[index].size = (out_h, out_w)
        return datapoint
