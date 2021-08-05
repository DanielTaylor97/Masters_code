import numpy as np
import tensorflow as tf

################
# LINEAR RULES #
################

def next_linear_dense(DX, W, Mp, Mm):
    # Mp refers to Mpp and Mmp
    # Mm refers to Mpm and Mmm

    z = np.tile(np.matmul(DX, W), reps=[W.shape[0], 1])

    mp = (z > 0) * W
    mm = (z > 0) * W

    Mp_new = np.matmul(Mp, mp.T) + np.matmul(Mp, mm.T)
    Mm_new = np.matmul(Mm, mp.T) + np.matmul(Mm, mm.T)

    return Mp_new, Mm_new


def next_linear_bn(DX, W, Mp, Mm):
    # Mp refers to Mpp and Mmp
    # Mm refers to Mpm and Mmm

    print(DX.shape)
    z = DX * W
    print(z.shape)

    mp = (z > 0) * W
    mm = (z < 0) * W
    print(mp.shape)

    Mp_new = Mp * mp + Mp * mm
    Mm_new = Mm * mp + Mm * mm

    return Mp_new, Mm_new


def deepLIFT_gradprop_conv_2D(W, Rhat, stride=1):
    W_rev = np.flip(W, axis=(0, 1))
    W_rev = np.transpose(W_rev, axes=[0, 1, 3, 2])

    if stride != 1:
        bs = np.shape(Rhat)[0]
        x = np.shape(Rhat)[1]
        y = np.shape(Rhat)[2]
        fs = np.shape(Rhat)[3]
        Rhat_exp = np.zeros((bs, stride * x, stride * y, fs))
        Rhat_exp[:, 1::stride, 1::stride, :] = Rhat
        Rhat = Rhat_exp

    DX_t = tf.nn.conv2d(Rhat, W_rev, strides=[1, 1, 1, 1], padding='SAME')
    sess = tf.compat.v1.Session()
    with sess.as_default():
        DX = DX_t.numpy()

        return DX


def next_linear_conv_2D(DX, W, Mp, Mm, stride=1):
    Wp = (W > 0) * W
    Wm = (W < 0) * W

    DXp = DX > 0
    DXm = DX < 0

    mpp = deepLIFT_gradprop_conv_2D(Wp, Mp, stride)
    mmp = deepLIFT_gradprop_conv_2D(Wm, Mp, stride)
    mpm = deepLIFT_gradprop_conv_2D(Wp, Mm, stride)
    mmm = deepLIFT_gradprop_conv_2D(Wm, Mm, stride)

    Mp = DXp * mpp + DXm * mmp
    Mm = DXp * mmm + DXm * mpm

    return Mp, Mm


def deepLIFT_gradprop_conv_3D(W, Rhat, stride=1):
    W_rev = np.flip(W, axis=(0, 1, 2))
    W_rev = np.transpose(W_rev, axes=[0, 1, 2, 4, 3])

    if stride != 1:
        bs = np.shape(Rhat)[0]
        x = np.shape(Rhat)[1]
        y = np.shape(Rhat)[2]
        z = np.shape(Rhat)[3]
        fs = np.shape(Rhat)[4]
        Rhat_exp = np.zeros((bs, stride * x, stride * y, stride * z, fs))
        Rhat_exp[:, 1::stride, 1::stride, 1::stride, :] = Rhat
        Rhat = Rhat_exp

    DX_t = tf.nn.conv3d(Rhat, W_rev, strides=[1, 1, 1, 1, 1], padding='SAME')
    sess = tf.compat.v1.Session()
    with sess.as_default():
        DX = DX_t.numpy()

        return DX


def next_linear_conv_3D(DX, W, Mp, Mm, stride=1):
    # Mp refers to Mpp and Mmp
    # Mm refers to Mpm and Mmm
    Wp = (W > 0) * W
    Wm = (W < 0) * W

    DXp = DX > 0
    DXm = DX < 0

    mpp = deepLIFT_gradprop_conv_3D(Wp, Mp, stride)
    mmp = deepLIFT_gradprop_conv_3D(Wm, Mp, stride)
    mpm = deepLIFT_gradprop_conv_3D(Wp, Mm, stride)
    mmm = deepLIFT_gradprop_conv_3D(Wm, Mm, stride)

    Mp = DXp * mpp + DXm * mmp
    Mm = DXp * mmm + DXm * mpm

    return Mp, Mm


def deepLIFT_gradprop_pooling_2D(X, DY):
    ##X, S
    x = int(2 * np.ceil(X.shape[1] / 2.0))
    y = int(2 * np.ceil(X.shape[2] / 2.0))

    DX = np.zeros(shape=(X.shape[0], x, y, X.shape[-1]))

    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        DX[:, i:x:2, j:y:2, :] += DY * 0.25
    if x > X.shape[1]:
        DX[:, -2, :, :] += DX[:, -1, :, :]
        DX = np.delete(DX, -1, 1)
    if y > X.shape[2]:
        DX[:, :, -2, :] += DX[:, :, -1, :]
        DX = np.delete(DX, -1, 2)
    return DX


def next_linear_pooling_2D(DX, Mp, Mm):
    # Mp refers to Mpp and Mmp
    # Mm refers to Mpm and Mmm
    DXp = DX > 0
    DXm = DX < 0

    mp = deepLIFT_gradprop_pooling_2D(DX, Mp)
    mm = deepLIFT_gradprop_pooling_2D(DX, Mm)

    Mp = DXp * mp + DXm * mm
    Mm = DXm * mp + DXp * mm

    return Mp, Mm


def deepLIFT_gradprop_pooling_3D(X, DY):
    ##X, S
    x = int(2 * np.floor(X.shape[1] / 2.0))
    y = int(2 * np.floor(X.shape[2] / 2.0))
    z = int(2 * np.floor(X.shape[3] / 2.0))

    DX = np.zeros(shape=(X.shape[0], x, y, z, X.shape[-1]))

    for i, j, k in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        DX[:, i:x:2, j:y:2, k:z:2, :] += DY * 0.125

    if x > X.shape[1]:
        DX[:, -2, :, :, :] += DX[:, -1, :, :, :]
        DX = np.delete(DX, -1, 1)
    if y > X.shape[2]:
        DX[:, :, -2, :, :] += DX[:, :, -1, :, :]
        DX = np.delete(DX, -1, 2)
    if z > X.shape[3]:
        DX[:, :, :, -2, :] += DX[:, :, :, -1, :]
        DX = np.delete(DX, -1, 3)
    if x < X.shape[1]:
        DX = np.pad(DX, ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)), 'constant')
    if y < X.shape[2]:
        DX = np.pad(DX, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)), 'constant')
    if z < X.shape[3]:
        DX = np.pad(DX, ((0, 0), (0, 0), (0, 0), (0, 1), (0, 0)), 'constant')
    return DX


def next_linear_pooling_3D(DX, Mp, Mm):
    # Mp refers to Mpp and Mmp
    # Mm refers to Mpm and Mmm
    DXp = DX > 0
    DXm = DX < 0

    mp = deepLIFT_gradprop_pooling_3D(DX, Mp)
    mm = deepLIFT_gradprop_pooling_3D(DX, Mm)

    Mp = DXp * mp + DXm * mm
    Mm = DXm * mp + DXp * mm

    return Mp, Mm


def next_linear_global_ave_3D(DX, Mp, Mm):
    # Mp refers to Mpp and Mmp
    # Mm refers to Mpm and Mmm
    tot = DX.shape[1] * DX.shape[2] * DX.shape[3]
    W = 1 / tot

    DXp = DX > 0
    DXm = DX < 0

    mp = DXp * W
    mm = DXm * W

    Mp = np.expand_dims(Mp, axis=(1, 2, 3))
    Mp = np.tile(Mp, reps=[1, DX.shape[1], DX.shape[2], DX.shape[3], 1])
    Mm = np.expand_dims(Mm, axis=(1, 2, 3))
    Mm = np.tile(Mm, reps=[1, DX.shape[1], DX.shape[2], DX.shape[3], 1])

    Mp = Mp * mp + Mp * mm
    Mm = Mm * mp + Mm * mm

    return Mp, Mm


def next_linear_add(DX1, DX2, Mp, Mm):
    mp1 = DX1 > 0
    mm1 = DX1 < 0
    mp2 = DX2 > 0
    mm2 = DX2 < 0

    Mp1 = Mp * mp1 + Mp * mm1
    Mm1 = Mm * mp1 + Mm * mm1

    Mp2 = Mp * mp2 + Mp * mm2
    Mm2 = Mm * mp2 + Mm * mm2

    return Mp1, Mm1, Mp2, Mm2


#################
# RESCALE RULES #
#################

def rescale_relu(DX, Mp, Mm):
    m = DX > 0

    Mp = Mp * m
    Mm = Mm * m

    return Mp, Mm


def rescale_softplus(DX, Mp, Mm, beta=1):
    DY = softplus(DX, beta)

    DX += (DX == 0)

    m = DY / DX

    Mp = Mp * m
    Mm = Mm * m

    return Mp, Mm


#######################
# REVEAL-CANCEL RULES #
#######################
def softplus(X, beta):
    Y = np.log(1 + np.exp(beta * X)) / beta
    return Y


def revealcancel_softplus(DX, X0, Mp, Mm, beta=1):
    DXp = (DX > 0) * DX + 1e-9
    DXm = (DX < 0) * DX - 1e-9

    DYp = 0.5 * (softplus(X0 + DXp, beta) - softplus(X0, beta)) + 0.5 * (
                softplus(X0 + DXm + DXp, beta) - softplus(X0 + DXm, beta))
    DYm = 0.5 * (softplus(X0 + DXm, beta) - softplus(X0, beta)) + 0.5 * (
                softplus(X0 + DXm + DXp, beta) - softplus(X0 + DXp, beta))

    mp = DYp / DXp
    mm = DYm / DXm

    Mp = Mp * mp
    Mm = Mm * mm

    return Mp, Mm