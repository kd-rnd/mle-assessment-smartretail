from utils import *
import mmengine
import json_tricks as json
import mimetypes

import yaml
import cv2
import argparse


def main(configs, args, detector, pose_estimator, visualizer):
    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                os.path.basename(args.input))
    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    input_type = mimetypes.guess_type(args.input)[0].split('/')[0] 

    if input_type == 'video':
        cap = cv2.VideoCapture(args.input)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        start = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1
            if frame_idx %100 ==0:
                print(f'Done {frame_idx}/{total_frame} frame!')

            if not success:
                break

            # topdown pose estimation
            pred_instances = process_one_image(configs, frame, detector,
                                            pose_estimator, visualizer,
                                            0.001)

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))


            time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()
        print('Done! Time inference:',time.time()-start)

    elif input_type == 'image':
        start = time.time()
        pred_instances_list, vis_img= process_img(configs, args.input, detector,pose_estimator, visualizer)
        print('Done! Time inference:',time.time()-start)
        if output_file:
            img_vis = visualizer.get_image()
            cv2.imwrite(output_file,mmcv.rgb2bgr(img_vis))

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument(
        '--config', type=str, default='configs/config.yaml', help='config file')
    parser.add_argument(
        '--input', type=str, default='demo_img/frame0418.jpg', help='Image/Video file')
    parser.add_argument(
        '--output-root',
        type=str,
        default='result',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='whether to save predicted results')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')

    args = parser.parse_args()


    #load config
    configs = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # build model from configs
    detector, pose_estimator, visualizer = build_model(configs)

    main(configs, args, detector, pose_estimator, visualizer)
