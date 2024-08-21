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
from pathlib import Path

from utils import dict2namespace, namespace2dict
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/Template-LBBDM-video.yaml",
        help="Path to the config file",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-r",
        "--result_path",
        type=str,
        default="results",
        help="The directory to save results",
    )

    parser.add_argument(
        "-t", "--train", action="store_true", default=False, help="train the model"
    )
    parser.add_argument(
        "--sample_to_eval",
        action="store_true",
        default=False,
        help="sample for evaluation",
    )
    parser.add_argument(
        "--sample_at_start",
        action="store_true",
        default=False,
        help="sample at start(for debug)",
    )
    parser.add_argument(
        "--save_top",
        action="store_true",
        default=False,
        help="save top loss checkpoint",
    )

    parser.add_argument(
        "--gpu_ids", type=str, default="0", help="gpu ids, 0,1,2,3 cpu=-1"
    )
    parser.add_argument("--port", type=str, default="12355", help="DDP master port")

    parser.add_argument(
        "--resume_model", type=str, default=None, help="model checkpoint"
    )

    parser.add_argument("--frame0", type=str, default=None, help="previous frame")
    parser.add_argument("--frame1", type=str, default=None, help="next frame")
    parser.add_argument("--folder", type=str, default=None, help="folder of frames")

    parser.add_argument(
        "--resume_optim", type=str, default=None, help="optimizer checkpoint"
    )

    parser.add_argument(
        "--max_epoch", type=int, default=None, help="optimizer checkpoint"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="optimizer checkpoint"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
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

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


def interpolate(frame0, frame1, model):
    with torch.no_grad():

        out = model.sample(frame0, frame1)
        if isinstance(out, tuple):  # using ddim
            out = out[0]
        out = torch.clamp(out, min=-1.0, max=1.0)  # interpolated frame in [-1,1]
    return out


def interpolate_2_frames(frame0_path, frame1_path, model):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )  # outptu tensor in [-1,1]

    a = np.array(Image.open(frame0_path)).astype(np.float32) / (2**16-1) *2 - 1
    frame0 = cv2.merge((a, a, a))
    frame0 = torch.from_numpy(np.moveaxis(np.array(frame0), -1, 0)).cuda().unsqueeze(0)
    b = np.array(Image.open(frame1_path)).astype(np.float32) / (2**16-1) *2 - 1
    frame1 = cv2.merge((b, b, b))
    frame1 = torch.from_numpy(np.moveaxis(np.array(frame1), -1, 0)).cuda().unsqueeze(0)
    # frame0 = transform(a).cuda().unsqueeze(0)
    # frame1 = transform(b).cuda().unsqueeze(0)
    I4 = interpolate(frame0, frame1, model)
    I2 = interpolate(frame0, I4, model)
    I1 = interpolate(frame0, I2, model)
    I3 = interpolate(I2, I4, model)

    I6 = interpolate(I4, frame1, model)
    I7 = interpolate(I6, frame1, model)
    I5 = interpolate(I4, I6, model)

    frames = [frame0, I1, I2, I3, I4, I5, I6, I7]
    imlist = [frame.cpu().numpy() for frame in frames]
    imlist = unnorm(imlist)
    images = [(im.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8) for im in imlist]
    return images


def number_from_file_name(file_name):
    return int(file_name.split(".")[0].split("_")[-1])


def interpolate_whole_video(folder_path: Path, model, frame_range=(480, 500)):
    video_path = Path(f"./interpolated/{folder_path.name}.mp4")
    output_path = Path(f"./interpolated/{folder_path.name}")

    if not output_path.exists():
        output_path.mkdir(parents=True)
    file_list = os.listdir(folder_path)
    file_list.sort(
        key=number_from_file_name
    )  # Sort files by the number in the file name
    file_list = list(filter(lambda x: (
        (number_from_file_name(x) >= frame_range[0]) and
        (number_from_file_name(x) <= frame_range[1]) 
    ), file_list))

    count = 0
    for file_name1, file_name2 in zip(file_list[:-1], file_list[1:]):
        file_path1 = folder_path / file_name1
        file_path2 = folder_path / file_name2
        new_frames = interpolate_2_frames(file_path1, file_path2, model)
        for i, frame in enumerate(new_frames):
            Image.fromarray(frame).save(
                    f"{output_path}/{file_path1.stem}_{i}.png" #{file_path1.suffix}"
            )
        print(f"{count}/{len(file_list)}")

        count += 1

    create_video(folder_path, output_path, video_path)


def unnorm(lst):
    out = []
    for a in lst:
        out.append(a / 2 + 0.5)
    return out


import os


def sort_key(s):
    s1 = s.stem.split("_")[-2]
    s2 = s.stem.split("_")[-1]


    return int(s1), int(s2)


def create_video(folder_path, output_path, video_path, fps=16):
    files = [output_path / f for f in os.listdir(output_path)]
    # + [
    #     folder_path / f for f in os.listdir(folder_path)
    # ]
    sorted_files = sorted(files, key=sort_key)
    first_image = cv2.imread(str(sorted_files[0]))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width, _ = first_image.shape

    orig_video_path = video_path.with_stem(video_path.stem + "_orig")
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    orig_video = cv2.VideoWriter(str(orig_video_path), fourcc, fps, (width, height))
    image_0 = None
    for file in sorted_files:
        image = cv2.imread(str(file))
        _, n = sort_key(file)
        if n == 0:
            image_0 = image
        video.write(image)
        orig_video.write(image_0)

    
    video.release()
    orig_video.release()


def convert_images_to_video(input_folder, output_file, fps):
    # Get the list of image files in the input folder
    image_files = sorted(
        [
            f
            for f in os.listdir(input_folder)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
    )
    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = first_image.shape
    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # Specify the codec for the output video file
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    # Iterate over each image and write it to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)
    # Release the video writer and close the video file
    video.release()
    cv2.destroyAllWindows()


def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args
    model = LatentBrownianBridgeModel(nconfig.model)
    state_dict_pth = args.resume_model
    frame0_path = args.frame0
    frame1_path = args.frame1
    folder_path = args.folder
    model_states = torch.load(state_dict_pth, map_location="cpu")
    model.load_state_dict(model_states["model"])
    model.eval()
    model = model.cuda()
    if folder_path:
        interpolate_whole_video(Path(folder_path), model)
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )  # outptu tensor in [-1,1]
    # a = (np.array(Image.open(frame0_path)) / 256).astype(np.uint8)
    # a = cv2.merge((a, a, a))
    # b = (np.array(Image.open(frame1_path)) / 256).astype(np.uint8)
    # b = cv2.merge((b, b, b))
    # frame0 = transform(a).cuda().unsqueeze(0)
    # frame1 = transform(b).cuda().unsqueeze(0)
    # I4 = interpolate(frame0, frame1, model)
    # I2 = interpolate(frame0, I4, model)
    # I1 = interpolate(frame0, I2, model)
    # I3 = interpolate(I2, I4, model)

    # I6 = interpolate(I4, frame1, model)
    # I7 = interpolate(I6, frame1, model)
    # I5 = interpolate(I4, I6, model)

    # imlist = [
    #     frame0.cpu().numpy(),
    #     I1.cpu().numpy(),
    #     I2.cpu().numpy(),
    #     I3.cpu().numpy(),
    #     I4.cpu().numpy(),
    #     I5.cpu().numpy(),
    #     I6.cpu().numpy(),
    #     I7.cpu().numpy(),
    #     frame1.cpu().numpy(),
    # ]
    # imlist = unnorm(imlist)
    # count = 0
    # d = "interpolated"
    # try:
    #     os.makedirs(f"./{d}")
    # except:
    #     pass
    # for im in imlist:
    #     img = Image.fromarray((im.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8))
    #     img.save(f"./{d}/{count}.png")
    #     count += 1
    # # Provide the path to the input image folder, output video file, and desired FPS
    # input_folder = f"./{d}"
    # output_file = f"./{d}/video.mp4"
    # fps = 2  # Frames per second
    # convert_images_to_video(input_folder, output_file, fps)


if __name__ == "__main__":
    main()
