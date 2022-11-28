import os
import sys
import argparse
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# DeepSORT
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

# YOLO
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run_app(source="./samples/left10_small.mp4"):
    # path for results
    if not os.path.isdir('./results/'):
        os.mkdir('./results/')
    save_dir = os.path.join(os.getcwd(), "results")
    print("save dir:", save_dir)

    # load deepsort
    max_cosine_distance = 0.4
    nn_budget = None
    model_filename = './stored_models/deep_sort_inference.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Load yolo
    device = select_device("cpu")
    model = DetectMultiBackend('./stored_models/yolov5.pt', device=device, dnn=False, data="data/coco128.yaml", fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    img_size = check_img_size((640, 640), s=stride)

    dataset = LoadImages(source, img_size=img_size, stride=stride, auto=pt)
    batch_size = 1
    vid_path, vid_writer = [None] * batch_size, [None] * batch_size

    model.warmup(imgsz=(1 if pt else batch_size, 3, *img_size))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    frame_idx = 0

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        #inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        frame_idx = frame_idx + 1
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            print("stem", p.stem)
            print("dir", save_dir)
            save_path = os.path.join(save_dir, p.name)
            # txt_path = os.path.join(save_dir, p.stem)
            s += '%gx%g ' % im.shape[2:]
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            # annotator = Annotator(im0, line_width=3, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # DeepSORT -> Extracting Bounding boxes and its confidence scores.
                bboxes = []
                scores = []
                for *boxes, conf, cls in det:
                    bbox_left = min([boxes[0].item(), boxes[2].item()])
                    bbox_top = min([boxes[1].item(), boxes[3].item()])
                    bbox_w = abs(boxes[0].item() - boxes[2].item())
                    bbox_h = abs(boxes[1].item() - boxes[3].item())
                    box = [bbox_left, bbox_top, bbox_w, bbox_h]
                    bboxes.append(box)
                    scores.append(conf.item())

                # DeepSORT -> Getting appearence features of the object.
                features = encoder(im0, bboxes)
                # DeepSORT -> Storing all the required info in a list.
                detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]

                # DeepSORT -> Predicting Tracks.
                tracker.predict()
                tracker.update(detections)
                # track_time = time.time() - prev_time

                # DeepSORT -> Plotting the tracks.
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    # DeepSORT -> Changing track bbox to top left, bottom right coordinates
                    bbox = list(track.to_tlbr())

                    # DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
                    txt = 'id:' + str(track.track_id)
                    (label_width, label_height), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    org = tuple(map(int, [int(bbox[0]), int(bbox[1]) - baseline]))

                    cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
                    cv2.putText(im0, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            # save video
            if vid_path[i] != save_path:
                vid_path[i] = save_path
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                save_path = str(Path(save_path).with_suffix('.mp4'))
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(vid_writer[i])
            vid_writer[i].write(im0)


if __name__ == '__main__':
    run_app()
