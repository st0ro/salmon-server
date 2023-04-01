import tensorflow as tf
import numpy as np
import cv2
import box_utils
from scipy.special import expit
from ssd_config import priors, center_variance, size_variance
import time
from tqdm import tqdm
from norfair import Detection, Tracker
import collections


INPUT_VIDEO_FILE = "./poolin/ashort2 0.25.mp4"
OUTPUT_VIDEO_FILE = "./out/ashort2 0.25.mp4"
SUBMERGED_THRESHOLD = 0.3

BIG_BOX_THRESHOLD = 0.8
TIME_DEQUE_LEN = 30


def crop(image, box):
    height = image.shape[0]
    width = image.shape[1]
    y_1 = box[0] * height
    x_1 = box[1] * width
    y_2 = box[2] * height
    x_2 = box[3] * width
    x_center = int((x_1+x_2)/2)
    y_center = int((y_1+y_2)/2)

    x_size = int(abs(x_1-x_2)/2*1.2)
    y_size = int(abs(y_1-y_2)/2*1.2)

    x_min = max(x_center-x_size, 0)
    x_max = min(x_center+x_size, width-1)
    y_min = max(y_center-y_size, 0)
    y_max = min(y_center+y_size, height-1)
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def draw_rect(image, box, text, box_color=(255, 255, 255)):
    y_1 = box[0]
    x_1 = box[1]
    y_2 = box[2]
    x_2 = box[3]
    x_center = int((x_1+x_2)/2)
    y_center = int((y_1+y_2)/2)

    x_size = int(abs(x_1-x_2)/2)
    y_size = int(abs(y_1-y_2)/2)
    # draw a rectangle on the image
    cv2.rectangle(image, (x_center-x_size, y_center-y_size),
                  (x_center+x_size, y_center+y_size), box_color, 3)
    cv2.putText(image, f"{text}", (x_center-x_size, y_center-y_size - 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

def centroid(image, box):
    # copied from draw_rect, dont need to optimize since both wont be called
    height = image.shape[0]
    width = image.shape[1]
    y_1 = box[0] * height
    x_1 = box[1] * width
    y_2 = box[2] * height
    x_2 = box[3] * width
    x_center = int((x_1+x_2)/2)
    y_center = int((y_1+y_2)/2)
    return [x_center, y_center]

def decoder(predictions):
    if isinstance(predictions, list):
        predictions = np.concatenate(predictions, axis=0)
    conf = predictions[:, :, :predictions.shape[2] - 4]  # / 256.0
    locations = predictions[:, :, predictions.shape[2] - 4:]  # / 256.0
    confidences = expit(conf)
    boxes = box_utils.np_convert_locations_to_boxes(
        locations, priors, center_variance, size_variance)
    boxes = box_utils.np_center_form_to_corner_form(boxes)
    return boxes, confidences


print("Starting Runker")

print("Loading models")
detection_model = tf.keras.models.load_model('./det_model.h5', compile=False)
class_model = tf.keras.models.load_model('./cla_model.h5', compile=False)
class_labels = ["not submerged", "submerged"]
print("Models loaded")

print("Loading input file", INPUT_VIDEO_FILE)
vid_in = cv2.VideoCapture(INPUT_VIDEO_FILE)
vid_width = int(vid_in.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(vid_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_frames = int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
vid_fps = int(vid_in.get(cv2.CAP_PROP_FPS))
print("Opening output file", OUTPUT_VIDEO_FILE)
vid_out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, cv2.VideoWriter_fourcc(
    *'mp4v'), vid_fps, (vid_width, vid_height))

tracker = Tracker(distance_function='euclidean', distance_threshold=500)

print("Starting to process video")
pbar = tqdm(total=vid_frames)
while True:
    ret, frame = vid_in.read()
    if ret:
        reframe = cv2.resize(frame, (320, 320), cv2.INTER_NEAREST)

        t = time.time()
        input_arr = np.array([reframe])  # Convert single image to a batch.
        input_arr = tf.keras.applications.mobilenet.preprocess_input(input_arr)
        predictions = detection_model.predict(input_arr, verbose=False)
        boxes, confidence = decoder(predictions)
        confidence = [c[1] for c in confidence[0]]
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            boxes[0],
            confidence,
            100,
            iou_threshold=0.3,
            score_threshold=0.5,
            soft_nms_sigma=0.0,
            name=None
        )

        trackerDets = []

        for i in range(len(selected_indices)):
            box = boxes[0][selected_indices[i]]
            y_1 = box[0]
            x_1 = box[1]
            y_2 = box[2]
            x_2 = box[3]

            # Big box removal/skipping
            if x_2 - x_1 > BIG_BOX_THRESHOLD or y_2 - y_1 > BIG_BOX_THRESHOLD:
                continue

            submerged = False
            decider = 'none'
            confidence = 0

            #horizontal line test
            if y_1 > SUBMERGED_THRESHOLD and y_2 > SUBMERGED_THRESHOLD:
                submerged = True
                decider = 'line'
                confidence = 1
            else:
                crop_img = np.array([crop(frame, box)])
                input_arr = tf.keras.applications.mobilenet.preprocess_input(crop_img)
                input_arr = tf.image.resize(input_arr, (128, 128))
                prediction = class_model.predict(input_arr, verbose=False)
                class_index = np.argmax(prediction[0])
                submerged = bool(class_index == 1)
                decider = 'model'
                confidence = prediction[0][class_index]

            points = np.array([[x_1 * vid_width, y_1 * vid_height], [x_2 * vid_width, y_2 * vid_height]])
            detData = {
                'submerged': submerged,
                'decider': decider,
                'confidence': confidence
            }
            trackerDets.append(Detection(points, data=detData))

        tracked = tracker.update(detections=trackerDets)

        for trackedObj in tracked:
            # update timing tracks
            if not hasattr(trackedObj, 'data'):
                trackedObj.data = {
                    "first_submerged": 0,
                    "sub_deque": collections.deque(maxlen=TIME_DEQUE_LEN),
                    "submerged": False,
                    "time_submerged": 0
                }

            que = trackedObj.data["sub_deque"]
            que.append(1 if trackedObj.last_detection.data['submerged'] else 0)
            submergedRes = bool((sum(que)/len(que)) > 0.5)

            # first frame submerged
            if trackedObj.data['submerged'] == False and submergedRes:
                trackedObj.data["first_submerged"] = 0
                trackedObj.data["time_submerged"] = 0
            trackedObj.data["submerged"] = submergedRes
            timeSub = trackedObj.data["time_submerged"] + 1/30 if submergedRes else 0
            trackedObj.data["time_submerged"] = timeSub

            est = trackedObj.estimate
            box = [est[0][1], est[0][0], est[1][1], est[1][0]]
            col = (255, 0, 0)
            if submergedRes:
                if timeSub < 15:
                    col = (0, 180, 0)
                elif timeSub < 30:
                    col = (0, 255, 255)
                else:
                    col = (0, 0, 255)
            draw_rect(frame, box, f"{'Submerged' if submergedRes else 'Not Submerged'}, {trackedObj.data['time_submerged']:.4f}", col)

        # number of tracked objects
        cv2.putText(frame, f"{len(tracked)} tracked objects", (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # processing FPS
        inference_time = time.time()-t
        cv2.putText(frame, f"{1 / inference_time:.1f} FPS", (20, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw submerged threshold
        cv2.line(frame, (0, int(vid_height * SUBMERGED_THRESHOLD)), (vid_width, int(vid_height * SUBMERGED_THRESHOLD)), (255, 0, 0), 1)

        vid_out.write(frame)
        pbar.update(1)
    else:
        break

pbar.close()
vid_in.release()
vid_out.release()

print("Video processed successfully")
