import numpy as np
import tensorflow as tf

#########################
# DENSE LAYER FUNCTIONS #
#########################

def forward_first(X, W, B):
    return np.matmul(X, W) + B


def relprop_first(X, W, R):
    highest = np.max(X)
    lowest = np.min(X)

    V = np.maximum(0, W)
    U = np.minimum(0, W)
    L = X * 0 + lowest  # vector of the same number -- lower bound
    H = X * 0 + highest  # vector of the same number -- upper bound

    Z = np.matmul(X, W) - np.matmul(L, V) - np.matmul(H, U) + 1e-9
    S = R / Z
    R = X * np.dot(S, W.T) - L * np.dot(S, V.T) - H * np.dot(S, U.T)
    return R


def forward_next(X, W, B):
    return np.matmul(X, W) + B


def relprop_eps_rule(X, W, R):
    Z = np.matmul(X, W) + 1e-9
    S = np.divide(R, Z)
    C = np.matmul(S, W.T)
    R = X * C
    return R


def relprop_zpls_rule(X, W, R):
    V = np.maximum(0, W)
    Z = np.matmul(X, V) + 1e-9  # we divide by this later -- no division by zero!
    S = R / Z
    C = np.dot(S, V.T)
    R = X * C
    return R


def relprop_ab_rule(X, W, B, R, a, b):
    Xp = np.maximum(0, X)
    Xm = np.minimum(0, X)

    Wp = np.maximum(0, W)
    Wm = np.minimum(0, W)

    # Z = np.matmul(Xp, W)

    Bp = 0  # np.maximum(0, B)
    Zp = np.matmul(Xp, Wp) + np.matmul(Xm, Wm) + 1e-9
    Sp = np.divide(R, (Zp + Bp))
    Cpp = np.matmul(Sp, Wp.T)
    Cpm = np.matmul(Sp, Wm.T)

    # U = np.minimum(0, W)
    Bm = 0  # np.minimum(0, B)
    Zm = np.matmul(Xp, Wm) + np.matmul(Xm, Wp) - 1e-9
    Sm = np.divide(R, (Zm + Bm))
    Cmp = np.matmul(Sm, Wp.T)
    Cmm = np.matmul(Sm, Wm.T)

    Ra = Xp * Cpp + Xm * Cpm
    Rb = Xm * Cmp + Xp * Cmm

    R = a * Ra - b * Rb
    return R


#########################
# CONVOLUTION FUNCTIONS #
#########################

def forward_conv_2D(X, V, B, stride=1):
    Y_t = tf.nn.conv2d(X, V, strides=1, padding='SAME')
    sess = tf.compat.v1.Session()
    with sess.as_default():
        Y = Y_t.numpy()
        return Y + B


def gradprop_conv_2D(W, Rhat, stride=1):
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


def relprop_nextconv_2D(X, W, B, R, a, b, stride=1):
    Xp = np.maximum(X, 0)
    Xm = np.minimum(X, 0)

    Wp = np.maximum(W, 0)
    Wm = np.minimum(W, 0)

    Bp = 0  # np.maximum(0, B)
    Zp = forward_conv_2D(Xp, Wp, 0, stride) + forward_conv_2D(Xm, Wm, 0, stride) + 1e-9
    Sp = np.divide(R, Zp + Bp)
    Cpp = gradprop_conv_2D(Wp, Sp, stride)
    Cpm = gradprop_conv_2D(Wm, Sp, stride)

    if b > 0:
        Bm = 0  # np.minimum(0, B)
        Zm = forward_conv_2D(Xm, Wp, 0, stride) + forward_conv_2D(Xp, Wm, 0, stride) - 1e-9
        Sm = np.divide(R, Zm + Bm)
        Cmp = gradprop_conv_2D(Wp, Sm, stride)
        Cmm = gradprop_conv_2D(Wm, Sm, stride)
    else:
        Cmp = 0
        Cmm = 0

    Ra = Xp * Cpp + Xm * Cpm
    Rb = Xp * Cmm + Xm * Cmp

    R = a * Ra - b * Rb
    return R


def relprop_firstconv_2D(X, W, R, stride=1):
    highest = np.max(X)
    lowest = np.min(X)

    V = np.maximum(0, W)
    U = np.minimum(0, W)

    L = X * 0 + lowest
    H = X * 0 + highest

    S = forward_conv_2D(X, W, R * 0, stride) - forward_conv_2D(L, V, R * 0, stride) - forward_conv_2D(H, U, R * 0,
                                                                                                      stride) + 1e-9

    C = np.divide(R, S)
    R = X * gradprop_conv_2D(W, C, stride) - L * gradprop_conv_2D(V, C, stride) - H * gradprop_conv_2D(U, C, stride)
    return R


def forward_conv_3D(X, V, B, stride=1):
    Y_t = tf.nn.conv3d(X, V, strides=[1, stride, stride, stride, 1], padding='SAME')
    sess = tf.compat.v1.Session()
    with sess.as_default():
        Y = Y_t.numpy()
        return Y + B


def gradprop_conv_3D(W, Rhat, stride=1):
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


def relprop_nextconv_3D(X, W, B, R, a, b, stride=1):
    Xp = np.maximum(0, X)
    Xm = np.minimum(0, X)

    Wp = np.maximum(0, W)
    Wm = np.minimum(0, W)

    Bp = 0  # np.maximum(0, B)
    Zp = forward_conv_3D(Xp, Wp, 0, stride) + forward_conv_3D(Xm, Wm, 0, stride) + 1e-9
    Sp = np.divide(R, (Zp + Bp))
    Cpp = gradprop_conv_3D(Wp, Sp, stride=stride)
    Cpm = gradprop_conv_3D(Wm, Sp, stride=stride)

    if b > 0:
        Bm = 0  # np.minimum(0, B)
        Zm = forward_conv_3D(Xp, Wm, 0, stride) + forward_conv_3D(Xm, Wp, 0, stride) - 1e-9
        Sm = np.divide(R, (Zm + Bm))
        Cmp = gradprop_conv_3D(Wp, Sm, stride)
        Cmm = gradprop_conv_3D(Wm, Sm, stride)
    else:
        Cmm = 0
        Cmp = 0

    Ra = Xp * Cpp + Xm * Cpm
    Rb = Xm * Cmp + Xp * Cmm

    R = a * Ra - b * Rb
    return R


def relprop_firstconv_3D(X, W, R, stride=1):
    lowest = np.min(X)
    highest = np.max(X)

    V = np.maximum(0, W)
    U = np.minimum(0, W)

    L = X * 0 + lowest
    H = X * 0 + highest

    Z = forward_conv_3D(X, W, R * 0, stride) - forward_conv_3D(L, V, R * 0, stride) - forward_conv_3D(H, U, R * 0,
                                                                                                      stride) + 1e-9

    S = np.divide(R, Z)

    R = X * gradprop_conv_3D(W, S, stride) - L * gradprop_conv_3D(V, S, stride) - H * gradprop_conv_3D(U, S, stride)
    return R


#####################
# POOLING FUNCTIONS #
#####################

def forward_pooling_2D(X):
    Y_t = tf.nn.avg_pool2d(X, 2, 2, 'SAME')
    sess = tf.compat.v1.Session()
    with sess.as_default():
        Y = Y_t.numpy()
        return Y


def gradprop_pooling_2D(X, DY):
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


def relprop_pooling_2D(X, R, a, b):
    Xp = np.maximum(0, X)
    Xm = np.minimum(0, X)

    Zp = forward_pooling_2D(Xp) + 1e-9
    Sp = np.divide(R, Zp)
    Cp = gradprop_pooling_2D(Xp, Sp)

    if b > 0:
        Zm = forward_pooling_2D(Xm) - 1e-9
        Sm = np.divide(R, Zm)
        Cm = gradprop_pooling_2D(Xm, Sm)
    else:
        Cm = 0

    R = a * Xp * Cp - b * Xm * Cm
    return R


def forward_pooling_3D(X):
    Y_t = tf.nn.avg_pool3d(X, 2, 2, 'VALID')
    sess = tf.compat.v1.Session()
    with sess.as_default():
        Y = Y_t.numpy()
        return Y


def gradprop_pooling_3D(X, DY):
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


def relprop_pooling_3D(X, X_l, R, a, b):
    # Z = forward_pooling_3D(X)
    ###Padded at the end of the axis with the one repeat of the last slice
    # print(np.sum(R))
    Xp = np.maximum(0, X)
    Xm = np.minimum(0, X)

    Zp = forward_pooling_3D(Xp) + 1e-9
    Sp = np.divide(R, Zp)
    Cp = gradprop_pooling_3D(X, Sp)

    if b > 0:
        Zm = forward_pooling_3D(Xm) - 1e-9
        Sm = np.divide(R, Zm)
        Cm = gradprop_pooling_3D(X, Sm)
    else:
        Cm = 0

    R = a * Xp * Cp - b * Xm * Cm

    return R


def relprop_global_ave_3D(X, R, a, b):
    tot = X.shape[1] * X.shape[2] * X.shape[3]
    W = 1 / tot

    Xp = np.maximum(0, X)
    Xm = np.minimum(0, X)

    Zp = np.average(Xp, (1, 2, 3)) + 1e-9
    Zm = np.average(Xm, (1, 2, 3)) - 1e-9

    Cp = np.divide((W * R), Zp)
    Cm = np.divide((W * R), Zm)

    Ra = Xp * Cp
    Rb = Xm * Cm

    R = a * Ra - b * Rb

    return R


#######################
# BATCHNORM FUNCTIONS #
#######################
'''y_i = gamma_c(x_i - mu_c)/sqrt(var_c + eps) + beta_c'''


def forward_batchnorm(X, gamma, beta, mean, var):
    eps = 1e-9
    var += eps
    X_hat = (X - mean) / np.sqrt(var)
    Y = gamma * X_hat + beta
    return Y


# From Hui, Binder
def relprop_batchnorm(X, gamma, beta, mean, var, R):
    eps = 1e-9
    var += eps
    W = gamma / np.sqrt(var)
    B = beta - mean * W
    absz = np.abs(X * W)
    absb = np.abs(B)
    R = R * absz / (absz + absb + eps)
    return R


##################
# RELU FUNCTIONS #
##################

def forward_relu(X):
    Z = X > 0
    return np.multiply(X, Z)


def relprop_relu(R):
    return R


####################
# SOFTPLUS FUNCTIONS#
####################

def relprop_softplus(R):
    return R


############################
# ADDITION LAYER FUNCTIONS #
############################

def forward_add(X1, X2):
    return X1 + X2


def relprop_add(R, X1, X2, a, b):
    X1p = np.maximum(0, X1)
    X1m = np.minimum(0, X1)
    X2p = np.maximum(0, X2)
    X2m = np.minimum(0, X2)

    eps = 1e-9

    tot_p = forward_add(X1p, X2p) + eps
    tot_m = forward_add(X1m, X2m) - eps

    R1p = np.multiply(R, np.divide(X1p, tot_p))
    R1m = np.multiply(R, np.divide(X1m, tot_m))
    R2p = np.multiply(R, np.divide(X2p, tot_p))
    R2m = np.multiply(R, np.divide(X2m, tot_m))

    R1 = a * R1p - b * R1m
    R2 = a * R2p - b * R2m

    return R1, R2