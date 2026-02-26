import argparse
import os
import sys
import json
import torch
import numpy as np
from loguru import logger
from collections import defaultdict, OrderedDict
from pathlib import Path
import motmetrics as mmp

# Thêm đường dẫn để import các module của dự án
sys.path.append(os.getcwd())

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from trackers.ocsort_tracker.ocsort import OCSort

def make_parser():
    parser = argparse.ArgumentParser("OC-SORT Custom Evaluation")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", action="store_true", help="Fuse conv and bn for testing.")
    
    # Tracker args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="iou threshold for association")
    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track")
    parser.add_argument("--inertia", type=float, default=0.2, help="inertia of smooth")
    parser.add_argument("--deltat", type=int, default=3, help="deltat for smooth")
    parser.add_argument("--use_byte", action="store_true", help="use byte in tracking")
    
    return parser

def load_gt(json_file):
    """Load ground truth from COCO JSON format."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    gt = defaultdict(list)
    
    # Map image_id to file_name (or frame_id if available)
    img_map = {}
    for img in data['images']:
        # Giả sử file_name có dạng 'video_name/frame_0001.jpg' hoặc tương tự
        # Hoặc chúng ta dùng video_id để gom nhóm
        vid_id = img.get('video_id', 1)
        frame_id = img.get('frame_id', -1)
        if frame_id == -1:
             # Cố gắng parse frame_id từ file_name nếu chưa có
             import re
             match = re.search(r'frame_(\d+)', img['file_name'])
             if match:
                 frame_id = int(match.group(1))
             else:
                 frame_id = img['id'] # Fallback
        
        img_map[img['id']] = (vid_id, frame_id)

    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_map: continue
        
        vid_id, frame_id = img_map[img_id]
        track_id = ann.get('track_id', -1)
        bbox = ann['bbox'] # x, y, w, h
        
        # Format: frame, id, x, y, w, h, conf, -1, -1, -1
        # Conf cho GT là 1
        gt_line = [frame_id, track_id, bbox[0], bbox[1], bbox[2], bbox[3], 1.0, -1, -1, -1]
        gt[vid_id].append(gt_line)
    
    # Convert to numpy arrays
    for k in gt:
        gt[k] = np.array(gt[k])
        
    return gt

def run_inference(exp, args, model, dataloader):
    """Run inference and return tracking results."""
    results = defaultdict(list)
    tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, 
                     use_byte=args.use_byte, min_hits=args.min_hits, 
                     inertia=args.inertia, delta_t=args.deltat)
    
    # Reset tracker for each video? 
    # Với custom dataset, nếu dataloader trộn lẫn các video, ta cần cẩn thận.
    # Tuy nhiên, MOTDataset thường load tuần tự.
    # Để đơn giản, ta giả định 1 video hoặc xử lý theo video_id.
    
    # Vì OCSort là online tracker, nó cần state. 
    # Ta cần biết khi nào chuyển video để reset tracker.
    
    current_vid = None
    
    for i, (img, _, info, img_id) in enumerate(dataloader):
        # info: (height, width, frame_id, video_id, file_name)
        # img_id: tensor([id])
        
        tensor_img = img.unsqueeze(0)
        if args.device == "gpu":
            tensor_img = tensor_img.cuda()
            if args.fp16:
                tensor_img = tensor_img.half()
        
        # Info extraction
        # MOTDataset trả về info là tuple, nhưng qua DataLoader nó được batch lại thành list của tuple/tensor
        # Vì batch_size=1 (thường dùng cho test tracking), ta lấy phần tử đầu tiên
        frame_id = info[2].item()
        video_id = info[3].item()
        file_name = info[4][0]
        
        if current_vid != video_id:
            if current_vid is not None:
                logger.info(f"Finished video {current_vid}, starting {video_id}")
                # Reset tracker state if needed (OCSort object might need recreation or reset method)
                tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, 
                     use_byte=args.use_byte, min_hits=args.min_hits, 
                     inertia=args.inertia, delta_t=args.deltat)
            current_vid = video_id

        # Inference
        with torch.no_grad():
            outputs = model(tensor_img)
            outputs = postprocess(outputs, exp.num_classes, args.conf, args.nms)
        
        output = outputs[0] # Batch size 1
        
        if output is not None:
            output = output.cpu().numpy()
            # Format: x1, y1, x2, y2, obj_conf, class_conf, class_pred
            
            # OCSort update
            # img_info: (height, width) - kích thước ảnh gốc
            img_h, img_w = info[0].item(), info[1].item()
            
            # Scale boxes back to original image size
            scale = min(exp.test_size[0] / img_h, exp.test_size[1] / img_w)
            
            # OCSort expects: x1, y1, x2, y2, score
            dets = output[:, :5] 
            dets[:, :4] /= scale
            
            online_targets = tracker.update(dets, [img_h, img_w], [img_h, img_w])
            
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # Format: frame, id, x, y, w, h, conf, -1, -1, -1
                # Note: frame_id should be 1-based for MOTMetrics usually
                res_line = [frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score, -1, -1, -1]
                results[video_id].append(res_line)
        else:
            # No detections, update tracker with empty
            online_targets = tracker.update(np.empty((0, 5)), [info[0].item(), info[1].item()], [info[0].item(), info[1].item()])
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                res_line = [frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score, -1, -1, -1]
                results[video_id].append(res_line)

    # Convert to numpy
    for k in results:
        results[k] = np.array(results[k])
        
    return results

def main(args):
    exp = get_exp(args.exp_file, None)
    
    if args.conf is not None: exp.test_conf = args.conf
    if args.nms is not None: exp.nmsthre = args.nms
    if args.tsize is not None: exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()

    if args.ckpt is None:
        ckpt_file = os.path.join(exp.output_dir, exp.exp_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
        
    logger.info(f"Loading checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    
    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    # Get Dataloader
    # Force batch_size = 1 for tracking
    dataloader = exp.get_eval_loader(batch_size=1, is_distributed=False)
    
    logger.info("Running inference...")
    ts_results = run_inference(exp, args, model, dataloader)
    
    logger.info("Loading Ground Truth...")
    # Construct path to json file based on exp config
    # Assuming exp.val_ann is the filename in datasets/custom_dataset/annotations/
    data_dir = os.path.join(os.getcwd(), "datasets", exp.data_dir_name)
    json_file = os.path.join(data_dir, "annotations", exp.val_ann)
    
    gt_results = load_gt(json_file)
    
    logger.info("Calculating Metrics...")
    accs = []
    names = []
    
    # Compare
    # Note: video_ids in gt and ts must match. 
    # Our load_gt uses video_id from JSON (default 1).
    # Our run_inference uses video_id from dataset (default 1).
    
    for vid_id in gt_results.keys():
        if vid_id in ts_results:
            logger.info(f"Evaluating Video ID: {vid_id}")
            gt = gt_results[vid_id]
            ts = ts_results[vid_id]
            
            # motmetrics expects dataframe or numpy array
            # We use compare_to_groundtruth which handles numpy arrays if formatted correctly
            # But mmp.io.loadtxt returns pandas DataFrame. Let's try to match that format or use lower level API.
            
            # Using mmp.utils.compare_to_groundtruth requires DataFrames usually or dicts.
            # Let's construct the accumulator directly.
            
            acc = mmp.MOTAccumulator(auto_id=True)
            
            # Get all unique frames
            all_frames = sorted(list(set(gt[:, 0]) | set(ts[:, 0])))
            
            for frame in all_frames:
                gt_objs = gt[gt[:, 0] == frame]
                ts_objs = ts[ts[:, 0] == frame]
                
                gt_ids = gt_objs[:, 1].astype(int).tolist()
                ts_ids = ts_objs[:, 1].astype(int).tolist()
                
                # Boxes: x, y, w, h
                gt_boxes = gt_objs[:, 2:6].tolist()
                ts_boxes = ts_objs[:, 2:6].tolist()
                
                # Calculate IoU distance
                dists = mmp.distances.iou_matrix(gt_boxes, ts_boxes, max_iou=0.5)
                
                acc.update(gt_ids, ts_ids, dists)
                
            accs.append(acc)
            names.append(str(vid_id))
        else:
            logger.warning(f"Video ID {vid_id} found in GT but not in Tracking Results.")

    if not accs:
        logger.error("No common videos found between GT and Tracking Results!")
        return

    mh = mmp.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=mmp.metrics.motchallenge_metrics, generate_overall=True)
    
    str_summary = mmp.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mmp.io.motchallenge_metric_names
    )
    print("\n" + str_summary)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
