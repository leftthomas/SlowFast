import argparse
import json

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models import create_slowfast

from utils import num_classes, clip_duration, test_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--model_path', default='result/slow_fast.pth', type=str, help='Model path')
    parser.add_argument('--video_path', default='data/test/applauding/_V-dzjftmCQ_000023_000033.mp4', type=str,
                        help='Video path')

    opt = parser.parse_args()
    model_path, video_path = opt.model_path, opt.video_path
    slow_fast = create_slowfast(model_num_class=num_classes)
    slow_fast.load_state_dict(torch.load(model_path, 'cpu'))
    slow_fast = slow_fast.cuda().eval()
    with open('result/kinetics_classnames.json', 'r') as f:
        kinetics_classnames = json.load(f)

    # create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
    video_data = test_transform(video_data)
    inputs = [i.cuda() for i in video_data['video']]
    pred = slow_fast(inputs)

    # get the predicted classes
    pred_classes = pred.topk(k=5)[1]
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print('predicted labels: {}'.format(pred_class_names))
