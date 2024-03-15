# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.distributed as dist
from mmengine.dist.utils import get_dist_info, init_dist
from tqdm import tqdm
import cv2
from PIL import Image
import decord
from time import time
import pickle
from collections import OrderedDict
from glob2 import glob

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))

try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    # import mmpose  # noqa: F401
    from mmpose.apis import inference_topdown_batch, inference_topdown, init_model
    from mmpose.structures import PoseDataSample
    from mmpose.utils import adapt_mmdet_pipeline
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

anno_path = '/home/zpengac/har/PoseSegmentationMask/ntu120_hrnet.pkl'
default_det_config = 'demo/mmdetection_cfg/rtmdet_tiny_8xb32-300e_coco.py'
default_det_ckpt = (
    'logs/coco_final/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')
default_pose_config = 'configs/body_2d_keypoint/topdown_psm_flow/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192_inference.py'
default_pose_ckpt = (
    'logs/coco_final/best_coco_AP_epoch_180.pth')
default_flow_ckpt = (
    'logs/jhmdb_final5/best_PCK_epoch_100.pth')

def get_bboxes_from_skeletons(skls, H, W, padding=10):
    y_mins = np.min(skls[..., 1], axis=(0, -1)).astype(int)
    y_maxs = np.max(skls[..., 1], axis=(0, -1)).astype(int)
    x_mins = np.min(skls[..., 0], axis=(0, -1)).astype(int)
    x_maxs = np.max(skls[..., 0], axis=(0, -1)).astype(int)

    y_mins = np.clip(y_mins - padding, a_min=0, a_max=None)
    y_maxs = np.clip(y_maxs + padding, a_min=None, a_max=H)
    x_mins = np.clip(x_mins - padding, a_min=0, a_max=None)
    x_maxs = np.clip(x_maxs + padding, a_min=None, a_max=W)

    bboxes = np.stack([x_mins, y_mins, x_maxs, y_maxs], axis=-1)
    bboxes = np.expand_dims(bboxes, axis=1)
    bboxes = [x for x in bboxes]
    
    return bboxes

def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def detection_inference(model, frames, batch_size_det=16):
    # results = []
    # for frame in frames:
    #     result = inference_detector(model, frame)
    #     results.append(result)
    # print(len(frames), frames[0].shape)

    batches = [frames[i:i+batch_size_det] for i in range(0, len(frames), batch_size_det)]
    results = []
    for batch in batches:
        result = inference_detector(model, batch)
        results.extend(result)
    return results

def write_psm(save_path, joint_masks, body_mask=None, obj_mask=None, rescale_ratio=1.0):

    os.makedirs(save_path.rsplit('/', 1)[0], exist_ok=True)

    out_masks = joint_masks
    if body_mask is not None:
        out_masks = np.concatenate([out_masks, np.expand_dims(body_mask, axis=0)], axis=0)
    if obj_mask is not None:
        out_masks = np.concatenate([out_masks, np.expand_dims(obj_mask, axis=0)], axis=0)
    # out_masks = F.interpolate(out_masks.unsqueeze(0), scale_factor=1.0/rescale_ratio, \
    #                           mode='bilinear', align_corners=False).squeeze(0)
    h, w = out_masks.shape[-2:]
    out_masks = np.stack([cv2.resize(mask, dsize=(int(w/rescale_ratio), int(h/rescale_ratio)), interpolation=cv2.INTER_LINEAR) for mask in out_masks])
    out_masks = (out_masks * 255).astype(np.uint8)
    J, H, W = out_masks.shape

    nw = 4
    nh = int(np.ceil(J / nw))
    canvas = np.zeros((H * nh, W * nw), dtype=np.uint8)
    for i in range(J):
        x = (i % nw) * W
        y = (i // nw) * H
        canvas[y:y+H, x:x+W] = out_masks[i]
    canvas[-1, -1] = J
    canvas[-1, -2] = H
    canvas[-1, -3] = W
    
    Image.fromarray(canvas).save(save_path)

def write_psm_from_pose_sample(save_path, pose_sample: PoseDataSample, rescale_ratio=1.0):
    masks = pose_sample.pred_fields.heatmaps.detach().cpu()

    mask_body = (F.sigmoid(masks[0]) > 0.5).float()
    mask_body = mask_body.numpy()
    mask_body_raw = (F.sigmoid(masks[0]) > 0.5).float().numpy()
    mask_joints = (F.sigmoid(masks[1:-1]) > 0.5).float()
    mask_joints_raw = (F.sigmoid(masks[1:-1]) > 0.5).float().numpy()
    mask_flow = (F.sigmoid(masks[-1]) > 0.5).float()
    mask_flow = mask_flow.numpy()
    mask_flow_raw = (F.sigmoid(masks[-1]) > 0.5).float().numpy()
    mask_joints_neg = (torch.max(mask_joints, dim=0, keepdim=True)[0] < 0.5).float()
    mask_joint = torch.argmax(torch.cat([mask_joints_neg, mask_joints], dim=0), dim=0)
    mask_joint = mask_joint.numpy()
    mask_joints = mask_joints.numpy()

    write_psm(save_path, mask_joints_raw, mask_body_raw, mask_flow_raw, rescale_ratio=rescale_ratio)

def pose_inference(anno_in, model, frames, det_results, compress=False, batch_size_pose=16):
    anno = cp.deepcopy(anno_in)
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = num_person

    # if compress:
    kp, frame_inds = [], []
    data = list(zip(frames, det_results))
    batches = [data[i:i+batch_size_pose] for i in range(0, len(frames), batch_size_pose)]
    pose_samples = []
    for batch in batches:
        batch_frames, batch_det_results = zip(*batch)
        # print(len(batch_frames), len(batch_det_results))
        batch_pose_samples = inference_topdown_batch(model, batch_frames, batch_det_results, bbox_format='xyxy')
        pose_samples.extend(batch_pose_samples)
    # print(len(pose_samples))
    for i, pose_sample in enumerate(pose_samples):
        save_path = anno['filename'].replace('.avi', f'/{i:03d}.png').replace('multiview_action_videos', 'psm').replace('/v', '_v')
        write_psm_from_pose_sample(save_path, pose_sample, rescale_ratio=4.0)
    # else:
    # kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)
    # for i, (f, d) in enumerate(zip(frames, det_results)):
    #     # Align input format
    #     # d = [dict(bbox=x) for x in list(d)]
    #     pose_samples = inference_topdown(model, f, d, bbox_format='xyxy')

        # for j, pose_sample in enumerate(pose_samples):
        #     save_path = anno['filename'].replace('.avi', f'/{i:03d}_{j:03d}.png').replace('videos', 'psm')
            # write_psm_from_pose_sample(save_path, pose_sample, rescale_ratio=4.0)
    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    # parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    # parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local-rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    parser.add_argument('--compress', action='store_true', help='whether to do K400-style compression')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # assert args.out.endswith('.pkl')

    # print('Loading video list...')
    # lines = mrlines(args.video_list)
    # lines = [x.split() for x in lines]

    # * We set 'frame_dir' as the base name (w/o. suffix) of each video
    # assert len(lines[0]) in [1, 2]
    # if len(lines[0]) == 1:
    #     annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines]
    # else:
    #     annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]

    # with open(anno_path, 'rb') as f:
    #     annos = pickle.load(f)['annotations']
    # for anno in annos:
    #     anno['filename'] = f'/scratch/iotsense/har/ntu/nturgb+d_rgb/{anno["frame_dir"]}_rgb.avi'
    #     anno['bboxes'] = get_bboxes_from_skeletons(anno['keypoint'], anno['img_shape'][0], anno['img_shape'][1])

    fns = glob('/scratch/PI/cqf/har_data/nwucla/multiview_action_videos/*/*.avi')
    fns = sorted(fns)
    annos = [{'filename': fn} for fn in fns]

    print('Loading models...')
    if args.non_dist:
        my_part = annos
        os.makedirs(args.tmpdir, exist_ok=True)
    else:
        init_dist('pytorch', backend='nccl')
        rank, world_size = get_dist_info()
        if rank == 0:
            os.makedirs(args.tmpdir, exist_ok=True)
        dist.barrier()
        my_part = annos[rank::world_size]

    # assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')
    det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)
    pose_model = init_model(args.pose_config, args.pose_ckpt, 'cuda')
    flow_sd = torch.load(default_flow_ckpt, map_location='cpu')['state_dict']
    sd_flow_dec = OrderedDict()
    for k, v in flow_sd.items():
        if k.startswith('head.flow_dec'):
            sd_flow_dec[k.replace('head.flow_dec.', '')] = v
    sd_flow_head = OrderedDict()
    for k, v in flow_sd.items():
        if k.startswith('head.flow_head'):
            sd_flow_head[k.replace('head.flow_head.', '')] = v
    pose_model.head.flow_dec.load_state_dict(sd_flow_dec)
    pose_model.head.flow_head.load_state_dict(sd_flow_head)

    print('Start inference...')
    results = []
    for anno in tqdm(my_part):
        if not osp.exists(anno['filename']):
            continue
        if osp.exists(anno['filename'].replace('.avi', f'/000.png').replace('multiview_action_videos', 'psm').replace('/v', '_v')):
            continue
        frames = extract_frame(anno['filename'])
        
        det_results = detection_inference(det_model, frames, batch_size_det=16)
        # * Get detection results for human
        # det_results = [x[0] for x in det_results]
        for i, det_sample in enumerate(det_results):
            # * filter boxes with small scores
            res = det_sample.pred_instances.bboxes.cpu().numpy()
            scores = det_sample.pred_instances.scores.cpu().numpy()
            res = res[scores >= args.det_score_thr]
            # * filter boxes with small areas
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]
            det_results[i] = res

        n_frames = min(len(frames), len(det_results))
        frames = frames[:n_frames]
        det_results = det_results[:n_frames]

        frames_next = cp.deepcopy(frames)
        frames_next.pop(0)
        frames_next.append(frames[-1])
        frames = [np.concatenate([frames[i], frames_next[i]], axis=-1) for i in range(len(frames))] 

        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        anno = pose_inference(anno, pose_model, frames, det_results, compress=args.compress, batch_size_pose=16)
        anno.pop('filename')
        results.append(anno)

if __name__ == '__main__':
    main()
