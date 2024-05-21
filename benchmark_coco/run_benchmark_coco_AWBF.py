import numpy as np
import pandas as pd
import json
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from multiprocessing import Process, cpu_count, Manager

from agents import BoundingBoxAgent, ModelSpecificAgent, CoordinatorAgent, DataProcessingAgent
from blackboard import Blackboard

def get_coco_annotations_data(file_in):
    images = dict()
    with open(file_in) as json_file:
        data = json.load(json_file)
        for i in range(len(data['images'])):
            image_id = data['images'][i]['id']
            images[image_id] = data['images'][i]
    return images

def get_coco_score(csv_path, coco_gt_file):
    images = get_coco_annotations_data(coco_gt_file)
    s = pd.read_csv(csv_path, dtype={'img_id': np.str, 'label': np.str})

    out = np.zeros((len(s), 7), dtype=np.float64)
    out[:, 0] = s['img_id']
    ids = s['img_id'].astype(np.int32).values
    x1 = s['x1'].values
    x2 = s['x2'].values
    y1 = s['y1'].values
    y2 = s['y2'].values
    for i in range(len(s)):
        width = images[ids[i]]['width']
        height = images[ids[i]]['height']
        out[i, 1] = x1[i] * width
        out[i, 2] = y1[i] * height
        out[i, 3] = (x2[i] - x1) * width
        out[i, 4] = (y2[i] - y1) * height
    out[:, 5] = s['score'].values
    out[:, 6] = s['label'].values

    coco_gt = COCO(coco_gt_file)
    detections = out
    image_ids = list(set(detections[:, 0]))
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats
    print(coco_metrics)
    return coco_metrics, detections

def process_with_awbf(pred_filenames, coco_gt_file, weights):
    blackboard = Blackboard()

    res_boxes = dict()
    ref_ids = None
    for j in range(len(pred_filenames)):
        if weights[j] == 0:
            continue
        print(f"Read {pred_filenames[j]}...")
        s = pd.read_csv(pred_filenames[j], dtype={'img_id': np.str, 'label': np.str})
        s.sort_values('img_id', inplace=True)
        s.reset_index(drop=True, inplace=True)
        ids = s['img_id'].values
        unique_ids = sorted(s['img_id'].unique())
        if ref_ids is None:
            ref_ids = tuple(unique_ids)
        else:
            if ref_ids != tuple(unique_ids):
                print(f"Different IDs in ensembled CSVs! {len(ref_ids)} != {len(unique_ids)}")
                s = s[s['img_id'].isin(ref_ids)]
                s.sort_values('img_id', inplace=True)
                s.reset_index(drop=True, inplace=True)
                ids = s['img_id'].values
        preds = s[['x1', 'y1', 'x2', 'y2', 'score', 'label']].values
        single_res = dict()
        for i in range(len(ids)):
            id = ids[i]
            if id not in single_res:
                single_res[id] = []
            single_res[id].append(preds[i])
        for el in single_res:
            if el not in res_boxes:
                res_boxes[el] = []
            res_boxes[el].append(single_res[el])

    # Process each image with AWBF
    all_boxes = []
    all_scores = []
    all_labels = []
    all_ids = []

    for img_id in sorted(res_boxes.keys()):
        bbox_data = res_boxes[img_id]
        bb_agents = [BoundingBoxAgent(i, data, blackboard) for i, data in enumerate(bbox_data)]
        model_agent = ModelSpecificAgent('modelA', blackboard)
        coordinator = CoordinatorAgent(blackboard)

        # Simulate the workflow
        for agent in bb_agents:
            agent.analyze_and_propose()

        model_agent.adjust_boxes()
        coordinator.review_and_decide()

        # Collect final fused boxes
        fused_boxes = blackboard.read("final_fused_boxes")
        if fused_boxes:
            all_boxes.append(fused_boxes['boxes'])
            all_scores.append(fused_boxes['scores'])
            all_labels.append(fused_boxes['labels'])
            all_ids.extend([img_id] * len(fused_boxes['boxes']))

    # Format the results as a DataFrame
    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    res = pd.DataFrame({'img_id': all_ids, 'x1': all_boxes[:, 0], 'y1': all_boxes[:, 1], 'x2': all_boxes[:, 2], 'y2': all_boxes[:, 3], 'score': all_scores, 'label': all_labels})
    return res

def benchmark_awbf(pred_filenames, weights, coco_gt_file, get_score_init=True):
    if get_score_init:
        for bcsv in pred_filenames:
            print(f"Evaluating initial predictions from {bcsv}")
            get_coco_score(bcsv, coco_gt_file)

    print("Running AWBF fusion and evaluation...")
    fused_results = process_with_awbf(pred_filenames, coco_gt_file, weights)
    fused_results.to_csv("awbf_ensemble.csv", index=False)
    get_coco_score("awbf_ensemble.csv", coco_gt_file)

if __name__ == '__main__':
    coco_gt_file = 'instances_val2017.json'
    pred_filenames = [
        'predictions/EffNetB0-preds.csv',
        'predictions/EffNetB0-mirror-preds.csv',
        'predictions/EffNetB1-preds.csv',
        'predictions/EffNetB1-mirror-preds.csv',
        'predictions/EffNetB2-preds.csv',
        'predictions/EffNetB2-mirror-preds.csv',
        'predictions/EffNetB3-preds.csv',
        'predictions/EffNetB3-mirror-preds.csv',
        'predictions/EffNetB4-preds.csv',
        'predictions/EffNetB4-mirror-preds.csv',
        'predictions/EffNetB5-preds.csv',
        'predictions/EffNetB5-mirror-preds.csv',
        'predictions/EffNetB6-preds.csv',
        'predictions/EffNetB6-mirror-preds.csv',
        'predictions/EffNetB7-preds.csv',
        'predictions/EffNetB7-mirror-preds.csv',
        'predictions/DetRS-valid.csv',
        'predictions/DetRS-mirror-valid.csv',
        'predictions/DetRS_resnet50-valid.csv',
        'predictions/DetRS_resnet50-mirror-valid.csv',
        'predictions/yolov5x_tta.csv',
        # Add more files as needed
    ]
    weights = [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 5, 7, 7, 9, 9, 8, 8, 5, 5, 10]  # Example weights, adjust as needed

    benchmark_awbf(pred_filenames, weights, coco_gt_file, get_score_init=True)
A