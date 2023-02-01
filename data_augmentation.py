import torch
import torch.nn.functional as F
import torchvision.transforms
from math import sin, cos, pi
import numbers
import numpy as np
import random
from typing import Union
from PIL import Image
from torchvision.transforms import ColorJitter
import cv2

"""
    Data augmentation functions.

    There are some problems with torchvision data augmentation functions:
    1. they work only on PIL images, which means they cannot be applied to tensors with more than 3 channels,
       and they require a lot of conversion from Numpy -> PIL -> Tensor

    2. they do not provide access to the internal transformations (affine matrices) used, which prevent
       applying them for more complex tasks, such as transformation of an optic flow field (for which
       the inverse transformation must be known).

    For these reasons, we implement my own data augmentation functions
    (strongly inspired by torchvision transforms) that operate directly
    on Torch Tensor variables, and that allow to transform an optic flow field as well.
"""


def normalize_image_sequence_(sequence, key='frame'):
    images = torch.stack([item[key] for item in sequence], dim=0)
    mini = np.percentile(torch.flatten(images), 1)
    maxi = np.percentile(torch.flatten(images), 99)
    images = (images - mini) / (maxi - mini + 1e-5)
    images = torch.clamp(images, 0, 1)
    for i, item in enumerate(sequence):
        item[key] = images[i, ...]


def put_hot_pixels_in_voxel_(voxel, hot_pixel_range=1.0, hot_pixel_fraction=0.001):
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    for i in range(num_hot_pixels):
        voxel[..., :, y[i], x[i]] = random.uniform(-hot_pixel_range, hot_pixel_range)


def add_hot_pixels_to_sequence_(sequence, hot_pixel_std=1.0, max_hot_pixel_fraction=0.001):
    hot_pixel_fraction = random.uniform(0, max_hot_pixel_fraction)
    voxel = sequence[0]['events']
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    val = torch.randn(num_hot_pixels, dtype=voxel.dtype, device=voxel.device)
    val *= hot_pixel_std
    # need to do multiprocessing
    for item in sequence:
        for i in range(num_hot_pixels):
            item['events'][..., :, y[i], x[i]] += val[i]


def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
    noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
    if noise_fraction < 1.0:
        mask = torch.rand_like(voxel) >= noise_fraction
        noise.masked_fill_(mask, 0)
    return voxel + noise


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> torchvision.transforms.Compose([
        >>>     torchvision.transforms.CenterCrop(10),
        >>>     torchvision.transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, is_flow=False):
        for t in self.transforms:
            x = t(x, is_flow)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class CenterCrop(object):
    """Center crop the tensor to a certain size.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        w, h = x.shape[2], x.shape[1]
        th, tw = self.size
        assert (th <= h)
        assert (tw <= w)
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve
            # the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[:, i:i + th, j:j + tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RobustNorm(object):
    """
    Robustly normalize tensor
    """

    def __init__(self, low_perc=0, top_perc=95):
        self.top_perc = top_perc
        self.low_perc = low_perc

    @staticmethod
    def percentile(t, q):
        """
        Return the ``q``-th percentile of the flattened input tensor's data.
        
        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``.
           
        :param t: Input tensor.
        :param q: Percentile to compute, which must be between 0 and 100 inclusive.
        :return: Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        try:
            result = t.view(-1).kthvalue(k).values.item()
        except RuntimeError:
            result = t.reshape(-1).kthvalue(k).values.item()
        return result

    def __call__(self, x, is_flow=False):
        """
        """
        t_max = self.percentile(x, self.top_perc)
        t_min = self.percentile(x, self.low_perc)
        # print("t_max={}, t_min={}".format(t_max, t_min))
        if t_max == 0 and t_min == 0:
            return x
        eps = 1e-6
        normed = torch.clamp(x, min=t_min, max=t_max)
        normed = (normed - torch.min(normed)) / (torch.max(normed) + eps)
        return normed

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '(top_perc={:.2f}'.format(self.top_perc)
        format_string += ', low_perc={:.2f})'.format(self.low_perc)
        return format_string


class LegacyNorm(object):
    """
    Rescale tensor to mean=0 and standard deviation std=1
    """

    def __call__(self, x, is_flow=False):
        """
        Compute mean and stddev of the **nonzero** elements of the event tensor
        we do not use PyTorch's default mean() and std() functions since it's faster
        to compute it by hand than applying those funcs to a masked array
        """
        nonzero = (x != 0)
        num_nonzeros = nonzero.sum()
        if num_nonzeros > 0:
            mean = x.sum() / num_nonzeros
            stddev = torch.sqrt((x ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero.float()
            x = mask * (x - mean) / stddev
        return x

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


class RandomCrop(object):
    """Crop the tensor at a random location.
    """

    def __init__(self, size, preserve_mosaicing_pattern=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.preserve_mosaicing_pattern = preserve_mosaicing_pattern

    @staticmethod
    def get_params(x, output_size):
        w, h = x.shape[2], x.shape[1]
        th, tw = output_size
        if th > h or tw > w:
            raise Exception("Input size {}x{} is less than desired cropped \
                    size {}x{} - input tensor shape = {}".format(w, h, tw, th, x.shape))
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: this parameter does not have any effect
        Returns:
            Tensor: Cropped tensor.
        """
        i, j, h, w = self.get_params(x, self.size)

        if self.preserve_mosaicing_pattern:
            # make sure that i and j are even, to preserve the mosaicing pattern
            if i % 2 == 1:
                i = i + 1
            if j % 2 == 1:
                j = j + 1

        return x[:, i:i + h, j:j + w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomRotationFlip(object):
    """Rotate the image by angle.
    """

    def __init__(self, degrees, p_hflip=0.5, p_vflip=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    @staticmethod
    def get_params(degrees, p_hflip, p_vflip):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        angle_rad = angle * pi / 180.0

        M_original_transformed = torch.FloatTensor([[cos(angle_rad), -sin(angle_rad), 0],
                                                    [sin(angle_rad), cos(angle_rad), 0],
                                                    [0, 0, 1]])

        if random.random() < p_hflip:
            M_original_transformed[:, 0] *= -1

        if random.random() < p_vflip:
            M_original_transformed[:, 1] *= -1

        M_transformed_original = torch.inverse(M_original_transformed)

        M_original_transformed = M_original_transformed[:2, :].unsqueeze(dim=0)  # 3 x 3 -> N x 2 x 3
        M_transformed_original = M_transformed_original[:2, :].unsqueeze(dim=0)

        return M_original_transformed, M_transformed_original

    def __call__(self, x, is_flow=False):
        """
            x: [C x H x W] Tensor to be rotated.
            is_flow: if True, x is an [2 x H x W] displacement field, which will also be transformed
        Returns:
            Tensor: Rotated tensor.
        """
        assert (len(x.shape) == 3)

        if is_flow:
            assert (x.shape[0] == 2)

        M_original_transformed, M_transformed_original = self.get_params(self.degrees, self.p_hflip, self.p_vflip)
        affine_grid = F.affine_grid(M_original_transformed, x.unsqueeze(dim=0).shape)
        transformed = F.grid_sample(x.unsqueeze(dim=0), affine_grid, align_corners=False)

        if is_flow:
            # Apply the same transformation to the flow field
            A00 = M_transformed_original[0, 0, 0]
            A01 = M_transformed_original[0, 0, 1]
            A10 = M_transformed_original[0, 1, 0]
            A11 = M_transformed_original[0, 1, 1]
            vx = transformed[:, 0, :, :].clone()
            vy = transformed[:, 1, :, :].clone()
            transformed[:, 0, :, :] = A00 * vx + A01 * vy
            transformed[:, 1, :, :] = A10 * vx + A11 * vy

        return transformed.squeeze(dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', p_flip={:.2f}'.format(self.p_hflip)
        format_string += ', p_vlip={:.2f}'.format(self.p_vflip)
        format_string += ')'
        return format_string


class RandomFlip(object):
    """
    Flip tensor along last two dims
    """

    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, x, is_flow=False):
        """
        :param x: [... x H x W] Tensor to be flipped.
        :param is_flow: if True, x is an [... x 2 x H x W] displacement field, which will also be transformed
        :return Tensor: Flipped tensor.
        """
        assert (len(x.shape) >= 2)
        if is_flow:
            assert (len(x.shape) >= 3)
            assert (x.shape[-3] == 2)

        dims = []
        if random.random() < self.p_hflip:
            dims.append(-1)

        if random.random() < self.p_vflip:
            dims.append(-2)

        if not dims:
            return x

        flipped = torch.flip(x, dims=dims)
        if is_flow:
            for d in dims:
                idx = -(d + 1)  # swap since flow is x, y
                flipped[..., idx, :, :] *= -1
        return flipped

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += '(p_flip={:.2f}'.format(self.p_hflip)
        format_string += ', p_vlip={:.2f})'.format(self.p_vflip)
        return format_string


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht), indexing='ij')
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid


class Flow_event_mask_Augmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def eraser_transform_e(self, img1, img2, event, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        h_e, w_e, d_e = event.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            # mean_color_e = np.mean(event.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
                event[y0:y0 + dy, x0:x0 + dx, int(d_e / 2):] = 0
        return img1, img2, event

    def spatial_transform(self, img1, img2, flow, event, mask):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            event = cv2.resize(event, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            if flow is not None:
                flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                flow = flow * [scale_x, scale_y]
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                event = event[:, ::-1]
                mask = mask[:, ::-1]
                if flow is not None:
                    flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                event = event[::-1, :]
                mask = mask[::-1, :]
                if flow is not None:
                    flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        if flow is not None:
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        event = event[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        mask = mask[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        mask = np.expand_dims(np.array(mask).astype(np.uint8), -1)
        return img1, img2, flow, event, mask

    def __call__(self, img1, img2, flow, event, mask):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, event, mask = self.spatial_transform(img1, img2, flow, event, mask)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        if flow is not None:
            flow = np.ascontiguousarray(flow)
        event = np.ascontiguousarray(event)
        mask = np.ascontiguousarray(mask)
        return img1, img2, flow, event, mask


def center_crop(img1, img2, flow, event, mask, crop_size):
    ht, wd = img1.shape[:2]
    i = int(round((ht - crop_size[0]) / 2.))
    j = int(round((wd - crop_size[1]) / 2.))
    mask = np.expand_dims(np.array(mask).astype(np.uint8), -1)
    # if self.preserve_mosaicing_pattern:
    #     # make sure that i and j are even, to preserve
    #     # the mosaicing pattern
    #     if i % 2 == 1:
    #         i = i + 1
    #     if j % 2 == 1:
    #         j = j + 1
    if flow is None:
        return img1[i:i + crop_size[0], j:j + crop_size[1]], \
               img2[i:i + crop_size[0], j:j + crop_size[1]], \
               None, \
               event[i:i +crop_size[0], j:j +crop_size[1]], \
               mask[i:i +crop_size[0], j:j +crop_size[1]]
