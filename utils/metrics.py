__author__ = 'noel0714'

import numpy as np


def revNormalize(frame):
    frame = np.maximum(frame, 0)
    frame = np.minimum(frame, 1)

    denormalize = np.uint8(frame * 255)

    return denormalize


def getAxisTotal(gen_frames, gt_frames):
    # [batch, width, height] or [batch, width, height, channel]
    if gen_frames.ndim == 3:
        axis = (1, 2)
        total = gen_frames.shape[1] * gen_frames.shape[2]
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
        total = gen_frames.shape[1] * gen_frames.shape[2] * gen_frames.shape[3]

    return axis, total


def batch_mae_frame_float(gen_frames, gt_frames):
    axis, total = getAxisTotal(gen_frames, gt_frames)

    x = np.float32(gen_frames)
    y = np.float32(gt_frames)

    absolute = np.absolute(x - y)
    mae = np.sum(absolute, axis=axis) / total

    return np.mean(mae)


def batch_mse(gen_frames, gt_frames):
    axis, total = getAxisTotal(gen_frames, gt_frames)

    x = np.float32(gen_frames)
    y = np.float32(gt_frames)

    squared = np.square(x - y) ** 0.5
    mse = np.sum(squared, axis=axis) / total

    return np.mean(mse)


def batch_psnr(gen_frames, gt_frames):
    _, total = getAxisTotal(gen_frames, gt_frames)

    mse = batch_mse(gen_frames, gt_frames)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse) / total

    return np.mean(psnr)