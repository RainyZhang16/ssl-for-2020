# -*- coding:utf-8 -*-
'''
Data augmentations for remote sening imagery (may be with multiple bands).
Version 1.0 2019-12-06 12:36:24 by QiJi
'''
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


class Resize(object):
    """
    Resize image (maybe with mask).
    Args:
        sample (dict): {'image': image, 'label': mask} (both are ndarrays)
        insize (tuple): (h, w) of image (mask)
        mode: 'seg'-segmentation; 'cls'-classification
    """
    def __init__(self, insize, mode='cls'):
        self.mode = mode
        self.insize = insize

        if self.mode == 'cls':
            self.transform = self.single_resize
        elif self.mode == 'seg':
            self.transform = self.join_resize

    def single_resize(self, sample):
        image = sample['image']
        h, w = self.insize
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        sample['image'] = image
        return sample

    def join_resize(self, sample):
        image, mask = sample['image'], sample['label']
        h, w = self.insize
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['image'], sample['label'] = image, mask
        return sample

    def __call__(self, sample):
        return self.transform(sample)


class RandomCrop(object):
    """
    Randdom crop image (maybe with mask).
    Args:
        sample (dict): {'image': image, 'label': mask} (both are ndarrays)
        insize (tuple): (h, w) of image (mask)
        mode: 'seg'-segmentation; 'cls'-classification
    """
    def __init__(self, insize, mode='cls'):
        self.mode = mode
        self.insize = insize

        if self.mode == 'cls':
            self.transform = self.random_crop

        elif self.mode == 'seg':
            self.transform = self.random_crop_pair

    def random_crop(self, sample):
        image = sample['image']
        image = random_crop(image, self.insize)
        sample['image'] = image
        return sample

    def random_crop_pair(self, sample):
        image, mask = sample['image'], sample['label']
        image, mask = random_crop_pair(image, mask, self.insize)
        sample['image'], sample['label'] = image, mask
        return sample

    def __call__(self, sample):
        return self.transform(sample)


class RandomScaleAspctCrop(object):
    """
    Randdom crop image (maybe with mask).
    Args:
        sample (dict): {'image': image, 'label': mask} (both are ndarrays)
        insize (tuple): (h, w) of image (mask)
        mode: 'seg'-segmentation; 'cls'-classification
    """
    def __init__(self, insize, scale=(0.25, 0.75), ratio=(3./4., 4./3.), p=0.5, mode='cls'):
        self.mode = mode
        self.insize = insize
        self.scale = scale
        self.ratio = ratio
        self.p = p

        if self.mode == 'cls':
            self.transform = self.random_scale_aspct_crop

        elif self.mode == 'seg':
            self.transform = self.join_random_scale_aspct_crop

    def random_scale_aspct_crop(self, sample):
        image = sample['image']

        H, W = image.shape[:2]  # ori_height, ori_width
        area = H*W
        if np.random.random() < self.p:
            for attempt in range(3):
                target_area = np.random.uniform(*self.scale) * area
                aspect_ratio = np.random.uniform(*self.ratio)

                w = int(round(np.sqrt(target_area * aspect_ratio)))
                h = int(round(np.sqrt(target_area / aspect_ratio)))

                if np.random.random() < 0.5:
                    w, h = h, w

                if w < W and h < H:
                    i = np.random.randint(0, H - h)  # crop start point(row/y)
                    j = np.random.randint(0, W - w)  # crop start point(col/x)
                    sample['image'] = resized_crop(
                        image, i, j, h, w, self.insize, cv2.INTER_LINEAR)
                    return sample
        else:
            w, h = W, H
        # Fallback
        w, h = min(w, W), min(h, H)
        i, j = (H - w) // 2, (W - w) // 2
        sample['image'] = resized_crop(
            image, i, j, h, w, self.insize, cv2.INTER_LINEAR)
        return sample

    def join_random_scale_aspct_crop(self, sample):
        image, mask = sample['image'], sample['label']

        H, W = image.shape[:2]  # ori_height, ori_width
        area = H*W
        if np.random.random() < self.p:
            for attempt in range(3):
                target_area = np.random.uniform(*self.scale) * area
                aspect_ratio = np.random.uniform(*self.ratio)

                w = int(round(np.sqrt(target_area * aspect_ratio)))
                h = int(round(np.sqrt(target_area / aspect_ratio)))

                if np.random.random() < 0.5:
                    w, h = h, w

                if w < W and h < H:
                    i = np.random.randint(0, H - h)  # crop start point(row/y)
                    j = np.random.randint(0, W - w)  # crop start point(col/x)
                    sample['image'] = resized_crop(
                        image, i, j, h, w, self.insize, cv2.INTER_LINEAR)
                    sample['label'] = resized_crop(
                        mask, i, j, h, w, self.insize, cv2.INTER_NEAREST)
                    return sample
        else:
            w, h = W, H
        # Fallback
        w, h = min(w, W), min(h, H)
        i, j = (H - w) // 2, (W - w) // 2
        sample['image'] = resized_crop(
            image, i, j, h, w, self.insize, cv2.INTER_LINEAR)
        sample['label'] = resized_crop(
            mask, i, j, h, w, self.insize, cv2.INTER_NEAREST)
        return sample

    def __call__(self, sample):
        return self.transform(sample)


class SpaceAugment(object):
    """
    Space data augmentations for image sized of [HW] or [HWC] (maybe with mask),
    support single- or multi- band(s) imagery.
    Args:
        sample (dict): {'image': image, 'label': mask} (both are ndarrays)
        mode: 'seg'-segmentation; 'cls'-classification
    """
    def __init__(self,
                 shift_limit=(-0.0, 0.0),
                 scale_limit=(-0.0, 0.0),
                 rotate_limit=(-0.0, 0.0),
                 aspect_limit=(-0.0, 0.0),
                 p=0.5,
                 mode='cls'):
        self.mode = mode
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.aspect_limit = aspect_limit
        self.p = p

        if self.mode == 'cls':
            self.transform = self.single_transform
        elif self.mode == 'seg':
            self.transform = self.join_transform

    def join_transform(self, sample):
        image, mask = sample['image'], sample['label']

        # Join Random Filp
        f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]  # [1, 0, -1, 2, 2]
        if f != 2:
            image, mask = filp_array(image, f), filp_array(mask, f)

        # Join Random Roate (Only 0, 90, 180, 270)
        k = np.random.randint(0, 4)  # [0, 1, 2, 3]
        image = np.rot90(image, k, (1, 0))  # clockwise
        mask = np.rot90(mask, k, (1, 0))

        # Affine transformation
        image, mask = randomShiftScaleRotate(
            image, mask,
            shift_limit=self.shift_limit, scale_limit=self.scale_limit,
            rotate_limit=self.rotate_limit, aspect_limit=self.aspect_limit,
            p=self.p)

        sample['image'], sample['label'] = image, mask
        return sample

    def single_transform(self, sample):
        image = sample['image']

        # Join Random Filp
        f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]  # [1, 0, -1, 2, 2]
        if f != 2:
            image = filp_array(image, f)

        # Random Roate (Only 0, 90, 180, 270)
        k = np.random.randint(0, 4)  # [0, 1, 2, 3]
        image = np.rot90(image, k, (1, 0))  # clockwise

        # Affine transformation
        image = randomShiftScaleRotate(
            image,
            shift_limit=self.shift_limit, scale_limit=self.scale_limit,
            rotate_limit=self.rotate_limit, aspect_limit=self.aspect_limit,
            p=self.p)

        sample['image'] = image
        return sample

    def __call__(self, sample):
        return self.transform(sample)


class ColorAugment(object):
    """ ColorJitter data augmentations for normal RGB image sized of (H, W, C).
    """
    def __init__(self, p=0.5,
                 hue_shift_limit=(-30, 30),
                 sat_shift_limit=(-5, 5),
                 val_shift_limit=(-15, 15)):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.p = p

    def __call__(self, sample):
        image = sample['image']

        image = randomHueSaturationValue(
            image, self.hue_shift_limit, self.sat_shift_limit,
            self.val_shift_limit, self.p
        )

        sample['image'] = image
        return sample


class ColorAugmentMixData(object):
    """ ColorJitter data augmentations for compose image which may consist of RGB, NIR and SAR.
    Input image must be [HWC].
    """
    def __init__(self, p=0.5,
                 hue_shift_limit=(-30, 30),
                 sat_shift_limit=(-5, 5),
                 val_shift_limit=(-15, 15),
                 brightness=0.01,
                 contrast=0.01):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            image = sample['image']

            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
                band_num = 1
            elif len(image.shape) == 3:
                band_num = image.shape[2]
            else:
                raise TypeError("image and label should be [HW] or [HWC]")

            if band_num >= 3:
                # RGB first
                image[:, :, :3] = randomHueSaturationValue(
                    image[:, :, :3], self.hue_shift_limit, self.sat_shift_limit,
                    self.val_shift_limit, u=1)
                # Other data
                if band_num > 3:
                    for c in range(3, band_num):
                        image[:, :, c] = randomColorAugment(image[:, :, c], 0.01, 0.01)
            elif band_num < 3:
                # Other data
                for c in range(0, band_num):
                    image[:, :, c] = randomColorAugment(image[:, :, c], 0.01, 0.01)

            sample['image'] = image

        return sample


class RandomNoise(object):
    """ Randomly add a kind of noise on image sized of [HW] or [HWC] (maybe with mask),
    support single- or multi- band(s) imagery.
    Args:
        modes: (string list) a set of noise patterns that may be applied to the image,
            but only one at a time.
            - 'gaussian'  Gaussian-distributed additive noise.
            - 'localvar'  Gaussian-distributed additive noise, with specified
                        local variance at each point of `image`.
            - 'poisson'   Poisson-distributed noise generated from the data.
            - 'salt'      Replaces random pixels with 1.
            - 'pepper'    Replaces random pixels with 0 (for unsigned images) or
                        -1 (for signed images).
            - 's&p'       Replaces random pixels with either 1 or `low_val`, where
                        `low_val` is 0 for unsigned images or -1 for signed
                        images.
            - 'speckle'   Multiplicative noise using out = image + n*image, where
                        n is uniform noise with specified mean & variance.
        p: (float) Probability of diverse noises being applied.
    """
    def __init__(self, modes=['gaussian', 's&p'], p=0.5):
        self.p = p
        self.modes = modes

    def __call__(self, sample):
        if np.random.random() < self.p:
            sample['image'] = random_noise(sample['image'], mode=np.random.choice(self.modes))

        return sample


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    Args:
        sample (dict): {'image': image, 'label': mask} (both are ndarrays)
        p (float): probability that image should be converted to grayscale.
    Returns:
        ndarray: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is multi- channel: grayscale version is random single channel
            of input image and return the output image with same shape as input image.
    """
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, sample):
        image = sample['image']

        if len(image.shape) == 3:
            if np.random.random() < self.p:
                # Choice a band of org_img as grayscale image
                band_num = image.shape[2]
                gray_band = np.random.randint(0, band_num)
                image = image[:, :, gray_band]
                image = np.repeat(np.expand_dims(image, axis=2), band_num, axis=2)
                # image = np.tile(image, (1, 1, band_num))
        sample['image'] = image
        return sample


class ToTensor(object):
    """ Convert `numpy.ndarray` image sized of [H,W] or [H,W,C] in the range [0, 255]
    to a torch.FloatTensor tensor of shape [C, H, W] in the range [0.0, 1.0].
    Args:
        sample
            - if sample is a dict in format of {'image': image}, return dict.
            - if sample is a numpy.ndarray, return ndarray.
    """
    def __call__(self, sample):
        if isinstance(sample, dict):
            image = sample['image']
            if len(image.shape) < 3:
                image = image[:, :, np.newaxis]
            # [H,W,C] array -> [C,H,W] tensor
            image = torch.from_numpy(image.copy().transpose((2, 0, 1)))
            image = image.float().div_(255)
            sample['image'] = image
        elif isinstance(sample, np.ndarray):
            if len(image.shape) < 3:
                image = image[:, :, np.newaxis]
            sample = torch.from_numpy(sample.copy().transpose((2, 0, 1)))
            sample = sample.float().div_(255)
        else:
            raise TypeError(
                "Input should be {'image': image array} or image array. Got {}".format(
                    type(sample)))
        return sample


class ToTensor2(object):
    """ Convert image and (may be with) label to tensor.
    Args:
        sample = {'image': image, 'label': label}
    Return:
        {'image': img_tensor, 'label', lbl_tensor}, where img_tensor is a
        torch.FloatTensor of shape [C, H, W] in the range [0.0, 1.0],
        lbl_tensor a torch.LongTensor.
    """
    def __call__(self, sample):
        img = sample['image'].copy()

        if isinstance(img, np.ndarray):
            if len(img.shape) < 3:
                img = img[:, :, np.newaxis]
            # [H,W,C] array -> [C,H,W] tensor
            img = torch.from_numpy(img.copy().transpose((2, 0, 1)))
            img = img.float().div_(255)
        else:
            raise TypeError(
                "Input image should be ndarray, got {}".format(type(img)))
        sample['image'] = img

        if 'label' in sample:
            lbl = sample['label']
            if isinstance(lbl, np.ndarray):
                sample['label'] = torch.from_numpy(lbl).long()
            else:
                sample['label'] = torch.tensor(lbl).long()

        return sample


class Normalizer(object):
    """
    Normalize image which is a Tensor of size (C, H, W), C maybe more than three!
    Args:
        Args:
        sample (dict): {'image': image, 'label': label},
        mean (sequence): Sequence of means for each channel (R,G,B,NIR, SAR).
        std (sequence): Sequence of standard deviations for each channely.
    """
    def __init__(self, mean, std):
        if mean is None:
            self.mean = [0.5, 0.5, 0.5]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.3125, 0.3125, 0.3125]
        else:
            self.std = std

    def __call__(self, sample):
        if isinstance(sample, dict):
            for t, m, s in zip(sample['image'], self.mean, self.std):
                t.sub_(m).div_(s)
        elif isinstance(sample, np.ndarray):
            for t, m, s in zip(sample, self.mean, self.std):
                t.sub_(m).div_(s)
        else:
            raise TypeError(
                "Input should be {'image': image array} or image array. Got {}".format(
                    type(sample)))
        return sample


def filp_array(array, flipCode):
    '''Filp an [HW] or [HWC] array vertically or horizontal according to flipCode.'''
    if flipCode != -1:
        array = np.flip(array, flipCode)
    elif flipCode == -1:
        array = np.flipud(array)
        array = np.fliplr(array)
    return array


def randomShiftScaleRotate(image,
                           mask=None,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT,
                           p=0.5):
    """
    Random shift scale rotate image (support multi-band) may be with mask.
    Args:
        p (float): Probability of rotation.
    """
    if np.random.random() < p:
        if len(image.shape) > 2:
            height, width, channel = image.shape
        else:  # TODO: test
            (height, width), channel = image.shape, 1

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect**0.5)
        sy = scale / (aspect**0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        if channel > 3:
            for c in range(channel):
                band = image[:, :, c]
                image[:, :, c] = cv2.warpPerspective(
                    band, mat, (width, height),
                    flags=cv2.INTER_LINEAR, borderMode=borderMode)
        else:
            image = cv2.warpPerspective(
                image, mat, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=borderMode)
        if mask is not None:
            mask = cv2.warpPerspective(
                mask, mat, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=borderMode)
    if mask is not None:
        return image, mask
    else:
        return image


def randomHueSaturationValue(image,
                             hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255),
                             u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0],
                                      hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return image


def resized_crop(image, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
    '''Crop the given PIL Image and resize it to desired size.
    Args:
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size: (Height, Width) must be tuple
    '''
    image = image[i:i+h, j:j+w]
    image = cv2.resize(image, size[::-1], interpolation)
    return image


def random_crop(array, crop_hw=(256, 256)):
    '''
    Crop image(label) randomly
    '''
    crop_h, crop_w = crop_hw
    if (crop_h < array.shape[0] and crop_w < array.shape[1]):
        x = np.random.randint(0, array.shape[0] - crop_h)  # row
        y = np.random.randint(0, array.shape[1] - crop_w)  # column
        return array[x:x + crop_h, y:y + crop_w]

    elif (crop_h == array.shape[0] and crop_w == array.shape[1]):
        return array

    else:
        raise Exception('Crop size > image.shape')


def random_crop_pair(image, label, crop_hw=(256, 256)):
    '''
    Crop image and label randomly
    '''
    crop_h, crop_w = crop_hw
    if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        raise Exception('Image and label must have the same shape')
    if (crop_h < image.shape[0] and crop_w < image.shape[1]):
        x = np.random.randint(0, image.shape[0] - crop_h)  # row
        y = np.random.randint(0, image.shape[1] - crop_w)  # column
        # label maybe multi-channel[H,W,C] or one-channel [H,W]
        return image[x:x + crop_h, y:y + crop_w], label[
            x:x + crop_h, y:y + crop_w]
    elif (crop_h == image.shape[0] and crop_w == image.shape[1]):
        return image, label
    else:
        raise Exception('Crop size > image.shape')


def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    """
    Function to add random noise of various types to a image.
    Parameters
    ----------
    image : ndarray
        Input image data sized of [HW] or [HWC] (support multi-band), from range of (0~255).
    mode : str, optional
        One of the following strings, selecting the type of noise to add:
        - 'gaussian'  Gaussian-distributed additive noise.
        - 'localvar'  Gaussian-distributed additive noise, with specified
                      local variance at each point of `image`.
        - 'poisson'   Poisson-distributed noise generated from the data.
        - 'salt'      Replaces random pixels with 1.
        - 'pepper'    Replaces random pixels with 0 (for unsigned images) or
                      -1 (for signed images).
        - 's&p'       Replaces random pixels with either 1 or `low_val`, where
                      `low_val` is 0 for unsigned images or -1 for signed
                      images.
        - 'speckle'   Multiplicative noise using out = image + n*image, where
                      n is uniform noise with specified mean & variance.
    seed : int, optional
        If provided, this will set the random seed before generating noise,
        for valid pseudo-random comparisons.
    clip : bool, optional
        If True (default), the output will be clipped after noise applied
        for modes `'speckle'`, `'poisson'`, and `'gaussian'`. This is
        needed to maintain the proper image data range. If False, clipping
        is not applied, and the output may extend beyond the range [0, 255].
    mean : float, optional
        Mean of random distribution. Used in 'gaussian' and 'speckle'.
        Default : 0.
    var : float, optional
        Variance of random distribution. Used in 'gaussian' and 'speckle'.
        Note: variance = (standard deviation) ** 2. Default : 0.01
    local_vars : ndarray, optional
        Array of positive floats, same shape as `image`, defining the local
        variance at every image point. Used in 'localvar'.
    amount : float, optional
        Proportion of image pixels to replace with noise on range [0, 1].
        Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.01
    salt_vs_pepper : float, optional
        Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
        Higher values represent more salt. Default : 0.5 (equal amounts)
    Returns
    -------
    out : ndarray
        Output floating-point image data on range [0, 255].
    Notes
    -----
    Speckle, Poisson, Localvar, and Gaussian noise may generate noise outside
    the valid image range. The default is to clip (not alias) these values,
    but they may be preserved by setting `clip=False`. Note that in this case
    the output may contain values outside the ranges [0, 255].
    Use this option with care.
    Because of the prevalence of exclusively positive floating-point images in
    intermediate calculations, it is not possible to intuit if an input is
    signed based on dtype alone. Instead, negative values are explicitly
    searched for. Only if found does this function assume signed input.
    Unexpected results only occur in rare, poorly exposes cases (e.g. if all
    values are above 50 percent gray in a signed `image`). In this event,
    manually scaling the input to the positive domain will solve the problem.
    The Poisson distribution is only defined for positive integers. To apply
    this noise type, the number of unique values in the image is found and
    the next round power of two is used to scale up the floating-point result,
    after which it is scaled back down to the floating-point image range.
    To generate Poisson noise against a signed image, the signed image is
    temporarily converted to an unsigned image in the floating point domain,
    Poisson noise is generated, then it is returned to the original range.
    """
    mode = mode.lower()

    if seed is not None:
        np.random.seed(seed=seed)

    allowedtypes = {
        'gaussian': 'gaussian_values',
        'localvar': 'localvar_values',
        'poisson': 'poisson_values',
        'salt': 'sp_values',
        'pepper': 'sp_values',
        's&p': 's&p_values',
        'speckle': 'gaussian_values'}

    kwdefaults = {
        'mean': 0.,
        'var': 0.01,
        'amount': 0.01,
        'salt_vs_pepper': 0.5,
        'local_vars': np.zeros_like(image) + 0.01}

    allowedkwargs = {
        'gaussian_values': ['mean', 'var'],
        'localvar_values': ['local_vars'],
        'sp_values': ['amount'],
        's&p_values': ['amount', 'salt_vs_pepper'],
        'poisson_values': []}

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[mode]]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowedkwargs[allowedtypes[mode]]))

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[mode]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    img_type = image.dtype
    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)
        out = image + (noise*128).astype(img_type)

    elif mode == 'localvar':
        # Ensure local variance input is correct
        if (kwargs['local_vars'] <= 0).any():
            raise ValueError('All values of `local_vars` must be > 0.')

        # Safe shortcut usage broadcasts kwargs['local_vars'] as a ufunc
        noise = np.random.normal(0, kwargs['local_vars'] ** 0.5)
        out = image + (noise * 128).astype(img_type)

    elif mode == 'poisson':
        # Determine unique values in image & calculate the next power of two
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))

        # Generating noise for each unique value in image.
        out = (np.random.poisson(image * vals) / float(vals)).astype(img_type)

    elif mode == 'salt':
        # Re-call function with mode='s&p' and p=1 (all salt noise)
        out = random_noise(image, mode='s&p', seed=seed,
                           amount=kwargs['amount'], salt_vs_pepper=1.)

    elif mode == 'pepper':
        # Re-call function with mode='s&p' and p=1 (all pepper noise)
        out = random_noise(image, mode='s&p', seed=seed,
                           amount=kwargs['amount'], salt_vs_pepper=0.)

    elif mode == 's&p':
        out = image.copy()
        p = kwargs['amount']
        q = kwargs['salt_vs_pepper']
        flipped = np.random.choice([True, False], size=image.shape,
                                   p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape,
                                  p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 255
        out[flipped & peppered] = 0

    elif mode == 'speckle':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)
        out = image + (image * noise).astype(img_type)

    # Clip back to original range, if necessary
    if clip:
        out = np.clip(out, 0, 255)

    return out


def randomColorAugment(image, brightness=0.1, contrast=0.1):
    if brightness > 0:
        brightness_factor = np.random.uniform(max(0, 1-brightness), 1+brightness)
        if brightness_factor > 1:
            alpha = brightness_factor - 1
            degenerate = np.ones(image.shape, dtype=np.uint8) * 255
        elif brightness_factor <= 1:
            alpha = 1 - brightness_factor
            degenerate = np.zeros(image.shape, dtype=np.uint8)
        image = cv2.addWeighted(degenerate, alpha, image, (1-alpha), 0)

    # Adjust contrast, saturation and hue reference: https://zhuanlan.zhihu.com/p/24425116
    if contrast > 0:
        contrast_factor = np.random.uniform(max(0, 1-contrast), 1+contrast)
        image = np.clip(image * contrast_factor, 0, 255)
    return image


def test():
    # from torchvision.transforms import Compose

    # Do test here
    img_path = 'C:/Users/HP/Desktop/GLCC-master/GLCC_self_supervised/utils/example/ex_0.jpg'

    # lbl_path = r'example/P0170.txt'
    img = cv2.imread(img_path)[:, :, ::-1]
    # lbl = cv2.imread(lbl_path)
    #plt.imshow(img)
    #plt.show()




    # sample = {'image': img}

    # output = RatioResize(sample, min_side=500)
    # transform = Compose([SpaceAugment(flip_x=0.5)])
    # out_img = transform(sample)
    # noise_img_gaussian = random_noise(img)

    '''
    def plotnoise(img, mode, r, c, i):
        plt.subplot(r, c, i)
        if mode is not None:
            gimg = random_noise(img, mode=mode)
            plt.imshow(gimg)
        else:
            plt.imshow(img)
        plt.title(mode)
        plt.axis("off")

    plt.figure(figsize=(18, 24))
    r = 4
    c = 2
    plotnoise(img, "gaussian", r, c, 1)
    plotnoise(img, "localvar", r, c, 2)
    plotnoise(img, "poisson", r, c, 3)
    plotnoise(img, "salt", r, c, 4)
    plotnoise(img, "pepper", r, c, 5)
    plotnoise(img, "s&p", r, c, 6)
    plotnoise(img, "speckle", r, c, 7)
    plotnoise(img, None, r, c, 8)
    plt.show()
    '''
    # Show result and self check

    pass


test()