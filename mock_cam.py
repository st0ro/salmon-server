import sys

if len(sys.argv) != 4:
    print("Mock cam usage: 'python mock_cam.py <server url> <video file> <submerged threshold>'")
    sys.exit()

import cv2
import tensorflow as tf
import time
from norfair import Detection, Tracker
import box_utils
from scipy.special import expit
from ssd_config import priors, center_variance, size_variance
import numpy as np
import base64
import requests

submergedThreshold = (float)(sys.argv[3])
server_url = sys.argv[1]
cam_id = requests.post(server_url + "/register/" + str(submergedThreshold)).text
print("Registered as Camera", cam_id)

print("Loading models")
detection_model = tf.keras.models.load_model('./det_model.h5', compile=False)
class_model = tf.keras.models.load_model('./cla_model.h5', compile=False)
class_labels = ["submerged", "not submerged"]
print("Models loaded")

print("Loading input file", sys.argv[2])
vid_in = cv2.VideoCapture(sys.argv[2])
vid_width = int(vid_in.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(vid_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_frames = int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
vid_fps = int(vid_in.get(cv2.CAP_PROP_FPS))

tracker = Tracker(distance_function='euclidean', distance_threshold=250)

def updateServer(frame, tracked, time, inference_time):
    payload = {}

    trackedlist = []
    for obj in tracked:
        est = obj.estimate
        trackedlist.append({
            "id": obj.id,
            "x": est[0][0] / vid_width,
            "y": est[0][1] / vid_height,
            "width": (est[1][0] - est[0][0]) / vid_width,
            "height": (est[1][1] - est[0][1]) / vid_height,
            "submerged": obj.last_detection.data['submerged'],
            "decider": obj.last_detection.data['decider'],
            "time": obj.last_detection.data['time'],
            "time_submerged": 3
        })
    payload["tracked"] = trackedlist

    success, im_buf_arr = cv2.imencode(".jpg", frame)
    if success == False:
        print("image encoding failed")
    image64 = base64.b64encode(im_buf_arr).decode()
    payload["image"] = "data:image/png;base64, " + image64
    #payload["image"] = "base64 string here"

    payload["time"] = time
    payload["inference_time"] = inference_time

    requests.post(server_url + "/update/" + cam_id, json=payload)

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

lastFrameNum = 0
print("Starting to run Mock Cam")
startTime = time.time()

while True:
    frameTime = time.time()
    passedTime = (frameTime - startTime)
    passedFrames = (int)(passedTime * vid_fps)
    frameNum = passedFrames % vid_frames
    vid_in.set(cv2.CAP_PROP_POS_FRAMES, frameNum)

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
            iou_threshold=0.5,
            score_threshold=0.5,
            soft_nms_sigma=0.5,
            name=None
        )

        trackerDets = []
        
        for i in range(len(selected_indices)):
            box = boxes[0][selected_indices[i]]
            y_1 = box[0]
            x_1 = box[1]
            y_2 = box[2]
            x_2 = box[3]

            submerged = False
            decider = 'none'

            #horizontal line test
            if y_1 > submergedThreshold and y_2 > submergedThreshold:
                submerged = True
                decider = 'line'
            else:
                crop_img = np.array([crop(frame, box)])
                input_arr = tf.keras.applications.mobilenet.preprocess_input(crop_img)
                input_arr = tf.image.resize(input_arr, (128, 128))
                prediction = class_model.predict(input_arr, verbose=False)
                class_index = np.argmax(prediction[0])
                submerged = bool(class_index == 1)
                decider = 'model'

            points = np.array([[x_1 * vid_width, y_1 * vid_height], [x_2 * vid_width, y_2 * vid_height]])
            detData = {
                'submerged': submerged,
                'decider': decider,
                'time': frameTime,
            }
            trackerDets.append(Detection(points, data=detData))

        #tracked = tracker.update(detections=trackerDets, period=passedFrames - lastFrameNum)
        tracked = tracker.update(detections=trackerDets)

        inference_time = time.time()-t
        lastFrameNum = frameNum
        
        updateServer(frame, tracked, frameTime, inference_time)
    else:
        print("uh oh")
