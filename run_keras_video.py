import tensorflow as tf
import numpy as np
import cv2
import box_utils
from scipy.special import expit
from ssd_config import priors, center_variance, size_variance
import time
from tqdm import tqdm
from norfair import Detection, Tracker, draw_points, draw_boxes


INPUT_VIDEO_FILE = "./poolin/a3.mp4"
OUTPUT_VIDEO_FILE = "./out/a3 tracked.mp4"


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
    height = image.shape[0]
    width = image.shape[1]
    y_1 = box[0] * height
    x_1 = box[1] * width
    y_2 = box[2] * height
    x_2 = box[3] * width
    x_center = int((x_1+x_2)/2)
    y_center = int((y_1+y_2)/2)

    x_size = int(abs(x_1-x_2)/2)
    y_size = int(abs(y_1-y_2)/2)
    # draw a rectangle on the image
    cv2.rectangle(image, (x_center-x_size, y_center-y_size),
                  (x_center+x_size, y_center+y_size), box_color, 1)
    cv2.putText(image, f"{text}", (x_center-x_size, y_center-y_size - 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


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
class_labels = ["submerged", "not submerged"]
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

tracker = Tracker(distance_function='euclidean', distance_threshold=100)

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
            iou_threshold=0.5,
            score_threshold=0.5,
            soft_nms_sigma=0.5,
            name=None
        )

        #print(f"Time taken to detect: {time.time()-t} seconds")
        #new_img = cv2.resize(frame, (int(640/frame.shape[0]*frame.shape[1]), 640))
        new_img = frame
        points = []
        dets = []

        for i in range(len(selected_indices)):
            box = boxes[0][selected_indices[i]]
            conf = selected_scores[i]
            crop_img = np.array([crop(frame, box)])
            input_arr = tf.keras.applications.mobilenet.preprocess_input(
                crop_img)
            input_arr = tf.image.resize(input_arr, (128, 128))
            #prediction = class_model.predict(input_arr, verbose=False)
            #class_index = np.argmax(prediction[0])
            #display_string = f"person: {conf:.4f}   {class_labels[class_index]}: {prediction[0][class_index]:.4f}"
            class_index = 0
            display_string = "person"
            # draw_rect(new_img, box, display_string, (0, 0, 255)
            # if class_index == 0 else (255, 255, 255))

            # change to use bounding boxes
            cen = centroid(new_img, box)
            points = np.array([cen, cen])
            points2 = np.array([[box[1] * vid_width, box[0] * vid_height], [box[3] * vid_width, box[2] * vid_height]])
            #cv2.putText(new_img, np.array2string(points2), (20, 100),
                    #cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            dets.append(Detection(points2))

        tracked = tracker.update(detections=dets)
        draw_points(new_img, drawables=tracked)
        draw_boxes(new_img, drawables=tracked)
        cv2.putText(new_img, f"{len(tracked)} tracked objects", (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        inference_time = time.time()-t
        cv2.putText(new_img, f"{1 / inference_time:.1f} FPS", (20, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        vid_out.write(new_img)
        pbar.update(1)
    else:
        break

pbar.close()
vid_in.release()
vid_out.release()

print("Video processed successfully")
