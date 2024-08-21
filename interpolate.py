import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
from functools import partial
from pathlib import Path

from utils import dict2namespace, namespace2dict
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('-c', '--config', type=str, default='configs/Template-LBBDM-video.yaml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")

    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='sample at start(for debug)')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')

    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')

    parser.add_argument('--frame0', type=str, default=None, help='previous frame')
    parser.add_argument('--frame1', type=str, default=None, help='next frame')
    
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--xN', type=int, default=1, help='augmentation of the frame rate between two input frames. must be a power of 2 [2,4,8,16,...]')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    assert is_power_of_two(args.xN) , "xN-1 is the number of frames to interpolate between the two images. Therefore it must be a power of 2."

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config

def is_power_of_two(n):
    """
    Checks if a number is a power of 2.

    Parameters:
    n (int): The number to check.

    Returns:
    bool: True if n is a power of 2, False otherwise.
    """
    # Check that n is greater than 0 and use bitwise operations 
    return n > 0 and (n & (n - 1)) == 0

def interpolate(frame0,frame1,model):
    with torch.no_grad():

        out = model.sample(frame0,frame1)
        if isinstance(out, tuple): # using ddim
            out = out[0]
        out =  torch.clamp(out, min=-1., max=1.) # interpolated frame in [-1,1]
    return out

def interpolte_N_frames(frame0, frame1, model, N):
    interpolated_frames = [frame0, frame1]
    interpolated_frames.insert(1, interpolate(frame0, frame1, model))
    with torch.no_grad():
        while len(interpolated_frames) < N + 1:
            old_interpolated_frames = interpolated_frames.copy()
            for i in range(len(old_interpolated_frames) - 1):
                interpolated_frames.insert(2*i + 1, interpolate(old_interpolated_frames[i], old_interpolated_frames[i+1], model))
    return [i.cpu().numpy() for i in interpolated_frames]




def unnorm(lst):
    out = []
    for a in lst:
        out.append(a/2 + 0.5)
    return out

def sort_key(s):
    s1 = s.stem.split("_")[-2]
    s2 = s.stem.split("_")[-1]

    if s1.isdigit() and s2.isdigit():
        n1 = int(s1)
        n2 = int(s2)
    else:
        n1 = int(s2)
        n2 = 0

    return n1, n2

def create_video(folder_path, video_path, fps=2):
    files = [folder_path / f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]
    # + [
    #     folder_path / f for f in os.listdir(folder_path)
    # ]
    sorted_files = sorted(files)
    first_image = cv2.imread(str(sorted_files[0]))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width, _ = first_image.shape
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for file in sorted_files:
        image = cv2.imread(str(file))
        video.write(image)
    video.release()

def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args
    model = LatentBrownianBridgeModel(nconfig.model)
    state_dict_pth = args.resume_model
    frame0_path = args.frame0
    frame1_path = args.frame1
    model_states = torch.load(state_dict_pth, map_location='cpu')
    model.load_state_dict(model_states['model'])
    model.eval()
    model = model.cuda()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #outptu tensor in [-1,1]
    frame0 = Image.open(frame0_path)
    frame1 = Image.open(frame1_path)
    to_array = partial(np.array, dtype=np.float32)
    if frame0._mode == 'I;16':
        frame0 = to_array(frame0)
        frame0 = frame0/(32767.5) - 1.0
        frame0 = cv2.merge((frame0, frame0, frame0))
        frame0 = torch.from_numpy(np.moveaxis(np.array(frame0), -1, 0)).cuda().unsqueeze(0)
    if frame1._mode == 'I;16':
        frame1 = to_array(frame1)
        frame1 = frame1/(32767.5) - 1.0
        frame1 = cv2.merge((frame1, frame1, frame1))
        frame1 = torch.from_numpy(np.moveaxis(np.array(frame1), -1, 0)).cuda().unsqueeze(0)
    else:
        frame0 = transform(frame0).cuda().unsqueeze(0)
        frame1 = transform(frame1).cuda().unsqueeze(0)
    # I4 = interpolate(frame0,frame1,model)
    # I2 = interpolate(frame0,I4,model)
    # I1 = interpolate(frame0,I2,model)
    # I3 = interpolate(I2,I4,model)

    # I6 = interpolate(I4,frame1,model)
    # I7 = interpolate(I6,frame1,model) 
    # I5 = interpolate(I4,I6,model)

    # imlist = [frame0.cpu().numpy(),I1.cpu().numpy(),I2.cpu().numpy(),I3.cpu().numpy(),I4.cpu().numpy(),I5.cpu().numpy(),I6.cpu().numpy(),I7.cpu().numpy(),frame1.cpu().numpy()]
    imlist = interpolte_N_frames(frame0, frame1, model, args.xN)
    imlist = unnorm(imlist)
    count = 0
    try:
        os.makedirs(f'{args.result_path}')
    except:
        pass
    for im in imlist:
        img = Image.fromarray((im.squeeze().transpose(1,2,0)*(2**16-1) / 256).astype(np.uint8))
        img.save(f'{args.result_path}/{count}.png')
        count += 1

    create_video(Path(f'{args.result_path}/'), Path(f'{args.result_path}/video.mp4'))

if __name__ == "__main__":
    main()
 
