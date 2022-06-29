import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from model.extraction_tools import initialize_networks, compute_kpts_desc, create_result_dir
from model.config_files.keynet_configs import keynet_config

def extract_features():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Key.Net PyTorch + HyNet local descriptor.'
                                                 'It returns local features as:'
                                                 'kpts: Num_kpts x 4 - [x, y, scale, score]'
                                                 'desc: Num_kpts x 128')

    parser.add_argument('--list-images', type=str, help='File containing the image paths for extracting features.',
                        required=True)

    parser.add_argument('--root-images', type=str, default='',
                        help='Indicates the root of the directory containing the images.'
                       'The code will copy the structure and save the extracted features accordingly.')

    parser.add_argument('--method-name', type=str, default='keynet_hynet_default',
                        help='The output name of the method.')

    parser.add_argument('--results-dir', type=str, default='extracted_features/',
                        help='The output path to save the extracted keypoint.')

    parser.add_argument('--config-file', type=str, default='KeyNet_default_config',
                        help='Indicates the configuration file to load Key.Net.')

    parser.add_argument('--num-kpts', type=int, default=5000,
                        help='Indicates the maximum number of keypoints to be extracted.')

    parser.add_argument('--gpu-visible-devices', type=str, default='0',
                        help='Indicates the device where model should run')


    args = parser.parse_known_args()[0]

    # Set CUDA GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Read Key.Net model and extraction configuration
    conf = keynet_config[args.config_file]
    keynet_model, desc_model = initialize_networks(conf, device)

    # read image and extract keypoints and descriptors
    f = open(args.list_images, "r")
    # for path_to_image in f:
    lines = f.readlines()
    for idx_im in tqdm(range(len(lines))):
        tmp_line = lines[idx_im].split('\n')[0]
        im_path = os.path.join(args.root_images, tmp_line)

        xys, desc = compute_kpts_desc(im_path, keynet_model, desc_model, conf, device, num_points=args.num_kpts)

        result_path = os.path.join(args.results_dir, args.method_name, tmp_line)
        create_result_dir(result_path)

        np.save(result_path + '.kpt', xys)
        np.save(result_path + '.dsc', desc)

    print('{} feature extraction finished.'.format(args.method_name))


if __name__ == "__main__":
    extract_features()
