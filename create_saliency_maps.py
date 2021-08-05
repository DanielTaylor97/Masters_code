from __future__ import division
import numpy as np
from tensorflow.keras import models
from DeepLIFT_functions import *
from LRP_functions import *


def fetch_model_and_data(arch='ResNet', masking=False, give_info=False):

    if masking:
        one_ids = np.load('test_ids_masking.npy', allow_pickle=True)
        one_model = models.load_model('masking_{}_model_finetuning'.format(arch))
        data_file = np.load('test_scans_masking.npz', allow_pickle=True)
    else:
        one_ids = np.load('test_ids_GCN.npy', allow_pickle=True)
        one_model = models.load_model('MRI_GCN_{}_model_softplus'.format(arch))
        data_file = np.load('test_scans_GCN.npz', allow_pickle=True)

    one_data = np.array([])
    for item in data_file.files:
        one_data = data_file[item]

    print("\n\n\nTest data loaded at {} bytes\n\n\n".format(one_data.nbytes))
    one_data = np.reshape(one_data, newshape=(-1, 233, 189, 197, 1))

    if give_info:
        print('Data shape: {}'.format(np.shape(one_data)))
        print()
        one_model.summary()
        print()

        for i in range(12):
            layer = one_model.get_layer(index=i)
            config = layer.get_config()
            print(config)

    return one_model, one_data, one_ids


###########
# LRP CMP #
###########

def LRP_CMP_3D(R, activations, allParams, alpha=1, bnLRP=True):
    beta = alpha - 1

    n_layers = len(activations)
    param_pos = -1

    R1, R2 = np.array([0]), np.array([0])

    for i in range(n_layers - 1):
        pos = -i - 1
        layerName = model.layers[pos].name
        print('{}. {}'.format(n_layers + pos, layerName))
        if layerName.startswith('dense'):
            R = relprop_relu(R)
            R = relprop_eps_rule(activations[pos - 1], allParams[param_pos - 1], R)
            param_pos += -1
        elif layerName.startswith('flatten'):
            R = np.reshape(R, activations[pos - 1].shape)
        elif layerName.startswith('global'):
            R = relprop_global_ave_3D(activations[pos - 1], R, alpha, beta)
        elif layerName.startswith('ave'):
            R = relprop_pooling_3D(activations[pos - 1], activations[pos], R, alpha, beta)
        elif 'relu' in layerName:
            if model.layers[pos + 1].name.startswith('conv'):
                R1 = relprop_relu(R1)
            else:
                R = relprop_relu(R)
        elif 'Softplus' in layerName or 'activation' in layerName:
            if model.layers[pos + 1].name.startswith('conv'):
                R1 = relprop_softplus(R1)
            else:
                R = relprop_softplus(R)
        elif layerName.startswith('add'):
            R1, R2 = relprop_add(R, activations[pos - 2], activations[pos - 1], alpha, beta)
        elif layerName.startswith('batch'):
            '''
            gamma, beta, mean, var
            Remember to feed these in in 'reverse order'
            '''
            if bnLRP:
                if model.layers[pos - 3].name.startswith('input'):
                    R = relprop_batchnorm(activations[pos - 1], allParams[param_pos - 4], allParams[param_pos - 3],
                                          allParams[param_pos - 2], allParams[param_pos - 1], R)
                elif np.shape(allParams[param_pos - 9])[0] == 1:
                    R2 = relprop_batchnorm(activations[pos - 2], allParams[param_pos - 4], allParams[param_pos - 3],
                                           allParams[param_pos - 2], allParams[param_pos - 1], R2)
                elif np.shape(allParams[param_pos - 6]) == 3:
                    R1 = relprop_batchnorm(activations[pos - 2], allParams[param_pos - 4], allParams[param_pos - 3],
                                           allParams[param_pos - 2], allParams[param_pos - 1], R1)
                else:
                    R1 = relprop_batchnorm(activations[pos - 1], allParams[param_pos - 4], allParams[param_pos - 3],
                                           allParams[param_pos - 2], allParams[param_pos - 1], R1)
            param_pos += -4
        elif layerName.startswith('conv'):
            if model.layers[pos - 2].name.startswith('input'):
                R = relprop_firstconv_3D(activations[pos - 1], allParams[param_pos - 1], R, stride=1)
                # R = relprop_nextconv_3D(activations[pos - 1], allParams[param_pos - 1], 0, R, alpha, beta, stride=2)
            elif np.shape(allParams[param_pos - 1])[0] == 1:
                R2 = relprop_nextconv_3D(activations[pos - 5], allParams[param_pos - 1], 0, R2, alpha, beta)
            else:
                R1 = relprop_nextconv_3D(activations[pos - 1], allParams[param_pos - 1], 0, R1, alpha, beta)
                if not model.layers[pos + 1].name.startswith('conv'):
                    R = R1 + R2
            param_pos += -1
        else:
            print('Layer type <<{}>> not recognised.'.format(layerName))

    return R


############
# DeepLIFT #
############

def DeepLIFT_3D(R, deepLIFT_acts, comp_activations, allParams, bn_deepLIFT=True):
    Mp = (R > 0) * R
    Mm = (R < 0) * R

    n_layers = len(deepLIFT_acts)
    param_pos = -1

    Mp1, Mp2 = np.array([0]), np.array([0])
    Mm1, Mm2 = np.array([0]), np.array([0])

    for i in range(n_layers - 1):
        pos = -i - 1
        layerName = model.layers[pos].name
        print('{}. {}'.format(n_layers + pos, layerName))

        if layerName.startswith('dense'):
            Mp, Mm = rescale_relu(deepLIFT_acts[pos], Mp, Mm)
            Mp, Mm = next_linear_dense(deepLIFT_acts[pos - 1], allParams[param_pos - 1], Mp, Mm)
            param_pos += -1
        elif layerName.startswith('flatten'):
            Mp = np.reshape(Mp, newshape=deepLIFT_acts[pos - 1].shape)
            Mm = np.reshape(Mm, newshape=deepLIFT_acts[pos - 1].shape)
        elif layerName.startswith('global'):
            Mp, Mm = next_linear_global_ave_3D(deepLIFT_acts[pos - 1], Mp, Mm)
        elif layerName.startswith('ave'):
            Mp, Mm = next_linear_pooling_3D(deepLIFT_acts[pos - 1], Mp, Mm)
        elif 'relu' in layerName:
            if model.layers[pos + 1].name.startswith('conv'):
                Mp1, Mm1 = rescale_relu(Mp1, Mm1)
            else:
                Mp, Mm = rescale_relu(Mp, Mm)
        elif 'Softplus' in layerName or 'activation' in layerName:
            if model.layers[pos + 1].name.startswith('conv'):
                Mp1, Mm1 = revealcancel_softplus(deepLIFT_acts[pos - 1], comp_activations[pos - 1], Mp1, Mm1)
            else:
                Mp, Mm = revealcancel_softplus(deepLIFT_acts[pos - 1], comp_activations[pos - 1], Mp, Mm)
        elif layerName.startswith('add'):
            Mp1, Mm1, Mp2, Mm2 = next_linear_add(deepLIFT_acts[pos - 2], deepLIFT_acts[pos - 1], Mp, Mm)
        elif layerName.startswith('batch'):
            '''
            gamma, beta, mean, var
            Remember to feed these in in 'reverse order'
            '''
            if bn_deepLIFT == True:
                weights = allParams[param_pos - 4] / np.sqrt(allParams[param_pos - 1] + 1e-9)
                if model.layers[pos - 3].name.startswith('input'):
                    Mp, Mm = next_linear_bn(deepLIFT_acts[pos - 1], weights, Mp, Mm)
                elif np.shape(allParams[param_pos - 9])[0] == 1:
                    Mp2, Mm2 = next_linear_bn(deepLIFT_acts[pos - 2], weights, Mp2, Mm2)
                elif np.shape(allParams[param_pos - 6]) == 3:
                    Mp1, Mm1 = next_linear_bn(deepLIFT_acts[pos - 2], weights, Mp1, Mm1)
                else:
                    Mp1, Mm1 = next_linear_bn(deepLIFT_acts[pos - 1], weights, Mp1, Mm1)
            param_pos += -4
        elif layerName.startswith('conv'):
            if model.layers[pos - 2].name.startswith('input'):
                Mp, Mm = next_linear_conv_3D(deepLIFT_acts[pos - 1], allParams[param_pos - 1], Mp, Mm, stride=1)
                R = (Mp + Mm) * (deepLIFT_acts[pos - 1])
                break  ####
            elif np.shape(allParams[param_pos - 1])[0] == 1:
                Mp2, Mm2 = next_linear_conv_3D(deepLIFT_acts[pos - 5], allParams[param_pos - 1], Mp2, Mm2)
            else:
                Mp1, Mm1 = next_linear_conv_3D(deepLIFT_acts[pos - 1], allParams[param_pos - 1], Mp1, Mm1)
                if not model.layers[pos + 1].name.startswith('conv'):
                    Mp = Mp1 + Mp2
                    Mm = Mm1 + Mm2
            param_pos += -1
        else:
            print('Layer type <<{}>> not recognised.'.format(layerName))

    return R


masked = True
model, data, ids = fetch_model_and_data("ResNet", masking=masked)

allParams = model.get_weights()

total = data.shape[0]


def generate_maps(method, LRP_alpha=1):
    for i in range(total):
        method_label = method

        X = data[i:i + 1]

        #####################################
        # GET ACTIVATIONS FROM FORWARD PASS #
        #####################################

        layer_outputs = [layer.output for layer in model.layers][0:]        # Creates an instance of the output of each layer
        activation_model = models.Model(inputs=model.input,
                                        outputs=layer_outputs)              # Creates a model that will return these outputs, given the model input
        acts = activation_model.predict(X)                                  # Returns a list of Numpy arrays: one array per layer activation

        activations = [X, ]
        activations.extend(acts)

        R = activations[-1]

        if method == 'LRP':
            method_label = 'LRP_{}'.format(LRP_alpha)
            map = LRP_CMP_3D(R, activations, allParams, alpha=LRP_alpha)
        elif method == 'DeepLIFT':
            comp = activation_model.predict(np.zeros_like(X))
            comp_activations = [np.zeros_like(X), ]
            comp_activations.extend(comp)

            deepLIFT_acts = []
            for i in range(len(activations)):
                deepLIFT_acts.append(activations[i] - comp_activations[i])
            activations = []

            map = DeepLIFT_3D(R, deepLIFT_acts, comp_activations, allParams)
        else:
            print('Invalid method given')
            break

        if masked:
            np.savez_compressed('{}_masking/{}_masking_{}.npy'.format(method_label, ids[i], method_label), map)
        else:
            np.savez_compressed('{}_brain_age/{}_aging_{}.npy'.format(method_label, ids[i], method_label), map)
        print('\n\n\nCreated {} saliency maps of {}\n\n\n'.format(i + 1, total))


generate_maps('LRP', LRP_alpha=1)
generate_maps('LRP', LRP_alpha=2)
generate_maps('LRP', LRP_alpha=3)
generate_maps('DeepLIFT')
