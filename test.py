'''
test.py:    Modified version of the original example.py file.
            This file runs classification on the examples images.
'''

import argparse
import glob

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import numpy as np
from PIL import Image
import sononet
import torch
from torch.autograd import Variable


disclaimer = '''
This is a PyTorch implementation of SonoNet:

Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard
Scan Planes in 2D Freehand Ultrasound", arXiv preprint:1612.05601 (2016)

This repository is based on
  https://github.com/baumgach/SonoNet-weights
which provides a theano+lasagne implementation.
'''
print(disclaimer)

# Configuration parameters
crop_range = [(115, 734), (81, 874)]  # [(top, bottom), (left, right)]
input_size = [224, 288]
image_path = './example_images/*.tiff'
label_names = ['3VV',
               '4CH',
               'Abdominal',
               'Background',
               'Brain (Cb.)',
               'Brain (Tv.)',
               'Femur',
               'Kidneys',
               'Lips',
               'LVOT',
               'Profile',
               'RVOT',
               'Spine (cor.)',
               'Spine (sag.) ']


def imcrop(image, crop_range):
    """ Crop an image to a crop range """
    return image[crop_range[0][0]:crop_range[0][1],
                 crop_range[1][0]:crop_range[1][1], ...]


def prepare_inputs():
    input_list = []
    for filename in glob.glob(image_path):

        # prepare images
        image = imread(filename)  # read
        image = imcrop(image, crop_range)  # crop
        image = np.array(Image.fromarray(image).resize(input_size, resample=Image.BICUBIC))
        image = np.mean(image, axis=2)  # convert to gray scale

        # convert to 4D tensor of type float32
        image_data = np.float32(np.reshape(image,
                                           (1, 1, image.shape[0],
                                            image.shape[1])))

        # normalise images by substracting mean and dividing by standard dev.
        mean = image_data.mean()
        std = image_data.std()
        image_data = np.array(255.0 * np.divide(image_data - mean, std),
                              dtype=np.float32)
        # Note that the 255.0 scale factor is arbitrary
        # it is necessary because the network was trained
        # like this, but the same results would have been
        # achieved without this factor for training.

        input_list.append(image_data)

    return input_list


def main(args):
    # convert weights parameter for sononet
    if args.weights == 'default':
        weights = True
    elif args.weights == 'none':
        weights = False
    elif args.weights == 'random':
        weights = 0
    else:
        weights = args.weights

    print('Loading network')
    net = sononet.SonoNet(args.network_name, weights=weights)
    net.eval()

    if args.cuda:
        print('Moving to GPU:')
        torch.cuda.device(args.GPU_NR)
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        net.cuda()

    print("\nPredictions using {}:".format(args.network_name))
    input_list = prepare_inputs()
    for image, file_name in zip(input_list, glob.glob(image_path)):

        # Run inference
        if args.cuda:
            x = Variable(torch.from_numpy(image).cuda())
        else:
            x = Variable(torch.from_numpy(image))
        outputs = net(x)
        confidence, prediction = torch.max(outputs.data, 1)

        # True labels are obtained from file name.
        true_label = file_name.split('/')[-1][0:-5]
        print(" - {} (conf: {:.2f}, true label: {})"
              .format(label_names[prediction[0]],
                      confidence[0], true_label))

        if args.display_images:
            plt.imshow(np.squeeze(image), cmap='gray')
            plt.show()


if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser(description='SonoNet')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--network_name', type=str, default='SN64', help='SN16, SN32 or SN64')
    parser.add_argument('--display_images', type=bool, default=False, help='Whether or not to show the images during inference')
    parser.add_argument('--GPU_NR', type=int, default=0, help='Choose the device number of your GPU')
    parser.add_argument('--weights', type=str, default='default', help='Select weight initialization. \
                        "default": Load weights from default *.pth weight file. "none": No weights are initialized. \
                        "random": Standard random weight initialization. Or: Pass the path to your own weight file')

    args = parser.parse_args()
    main(args)
