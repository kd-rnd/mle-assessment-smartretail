from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmdet.apis import inference_detector, init_detector
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import os
import mmcv
import numpy as np
import time

def build_model(configs):
    # build detector
    detector = init_detector(
        configs['inference']['det_config'], configs['inference']['det_checkpoint'], device=configs['inference']['device'])
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        configs['inference']['pose_config'],
        configs['inference']['pose_checkpoint'],
        device=configs['inference']['device'],
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=configs['inference']['draw_heatmap']))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = configs['visualize']['radius']
    pose_estimator.cfg.visualizer.alpha = configs['visualize']['alpha']
    pose_estimator.cfg.visualizer.line_width = configs['visualize']['thickness']
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=configs['visualize']['skeleton_style'])
    
    return detector, pose_estimator, visualizer


def process_img(configs, img, detector,pose_estimator, visualizer):
    pred_instances = process_one_image(configs, img, detector,
                                        pose_estimator, visualizer)
    pred_instances_list = split_instances(pred_instances)

    vis_img = visualizer.get_image()
    vis_img= mmcv.rgb2bgr(vis_img)
    return pred_instances_list, vis_img


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args['inference']['det_cat_id'],
                                   pred_instance.scores > args['inference']['bbox_thr'])]
    bboxes = bboxes[nms(bboxes, args['inference']['nms_thr']), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args['inference']['draw_heatmap'],
            draw_bbox=args['visualize']['draw_bbox'],
            show_kpt_idx=args['visualize']['show_kpt_idx'],
            skeleton_style=args['visualize']['skeleton_style'],
            show=False,
            wait_time=show_interval,
            kpt_thr=args['visualize']['kpt_thr'])


    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)

