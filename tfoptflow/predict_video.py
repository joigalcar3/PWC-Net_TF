from __future__ import absolute_import, division, print_function
from copy import deepcopy
from skimage.io import imread
import time
import numpy as np
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import display_img_pairs_w_flows
import os
from skimage.color import rgba2rgb
from skimage import img_as_ubyte
from optflow import flow_to_img
import cv2


def predict_video_PWCNET(video_path, fps_in, fps_out, img_folder, img_start, img_step, ckpt_path, gpu_devices, controller,
                         **kwargs):
    """
    Function to obtain the PWC-NET predicted Optical Flow from a video given in frames
    :param video_path: directory where the OF video should be stored
    :param fps_in: the number of fps of the original video
    :param fps_out: the number of fps of the output video
    :param img_folder: the location where images of the original video are found
    :param img_start: the index of the starting image of the video (use to avoid flight transients)
    :param img_step: the number of frames back used to compute the optical flow with the current image
    :param ckpt_path: the location where the NN model is found
    :param gpu_devices: the devices chosen for the inference
    :param controller: controller device to put the model's variables on
    :param kwargs: other variables not listed above that may be used
    :return:
    """
    # Build a list of image pairs to process
    end_img = kwargs["end_img"] if "end_img" in kwargs.keys() else len(os.listdir(img_folder))
    skip_pairs = kwargs["skip_pairs"] if "skip_pairs" in kwargs.keys() else 1
    img_names = sorted(os.listdir(img_folder))[img_start:end_img:int(fps_in/fps_out)]
    img_pairs = []
    for i in range(img_step, len(img_names), skip_pairs):
        image_path1 = os.path.join(img_folder, img_names[i-img_step])
        image_path2 = os.path.join(img_folder, img_names[i])
        image1, image2 = imread(image_path1), imread(image_path2)
        if image1.shape[2] == 4:
            # convert the image from RGBA2RGB
            image1 = rgba2rgb(image1)
        if image2.shape[2] == 4:
            # convert the image from RGBA2RGB
            image2 = rgba2rgb(image2)
        img_pairs.append((img_as_ubyte(image1), img_as_ubyte(image2)))

    # Configure the model for inference, starting with the default options
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = ckpt_path
    nn_opts['batch_size'] = 1
    nn_opts['gpu_devices'] = gpu_devices
    nn_opts['controller'] = controller

    # We're running the PWC-Net-large model in quarter-resolution mode
    # That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    # The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
    # of 64. Hence, we need to crop the predicted flows to their original size
    nn_opts['adapt_info'] = (1, 436, 1024, 2)

    # Instantiate the model in inference mode and display the model configuration
    nn = ModelPWCNet(mode='test', options=nn_opts)
    nn.print_config()

    # Generate the predictions and display them
    # time_lst = []
    # counter = 0
    # for img_pair in img_pairs:
    #     start = time.time()
    #     pred_label = nn.predict_from_img_pairs([img_pair], batch_size=1, verbose=False)
    #     end = time.time()
    #     elapsed_time = end - start
    #     time_lst.append(elapsed_time)
    #     print(f"Elapsed time of iter {counter}: {elapsed_time}")
    #     counter += 1
    # print(f"Average elapsed time: {np.mean(time_lst)}")

    pred_labels = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)

    # Store video
    frameSize = img_pairs[0][0].shape[1::-1]
    filename = os.path.join(video_path, video_path.split("\\")[-1] + f"_PWC_s{img_start}_f{fps_out}_k{img_step}_255.avi")
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps_out, frameSize)

    # Convert the predictions to images
    for pre_label in pred_labels:
        flow_img = flow_to_img(pre_label)
        out.write(flow_img)
    out.release()
    # display_img_pairs_w_flows(img_pairs, pred_labels)


if __name__ == "__main__":
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_city_256_144"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_city_512_288"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_City_1024_576"
    img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_City_1024_576_2"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Sintel_clean_ambush"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\KITTI_2015"
    start_img = 0
    end_img = None
    img_step = 1
    fps_rate_in = 30
    fps_rate_out = 30
    skip_pairs = 1
    if "Sintel" in img_folder:
        start_img = 0
        fps_rate_in = 2
        fps_rate_out = 2
    elif "KITTI" in img_folder:
        start_img = 0
        img_step = 1
        fps_rate_in = 1
        fps_rate_out = 1
        skip_pairs = 2
    model_directory = './models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
    gpu_dev = ['/device:GPU:0']
    contr = '/device:GPU:0'

    video_storage_folder = os.path.join("D:\\AirSim simulator\\FDD\\Optical flow\\video_storage",
                                        img_folder.split("\\")[-1])

    predict_video_PWCNET(video_storage_folder, fps_rate_in, fps_rate_out, img_folder, start_img, img_step,
                         model_directory, gpu_dev, contr, end_img=end_img, skip_pairs=skip_pairs)
