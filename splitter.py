import cv2
from tqdm import tqdm

FILENAME = 'j2'
INLOC = './poolin/' + FILENAME + '.mp4/'
OUTLOC = './poolout/' + FILENAME + '/' + FILENAME + '_frame'
SCALE = 2/3

print("Running on", FILENAME)

vidcap = cv2.VideoCapture(INLOC)
vid_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success, image = vidcap.read()
count = 0
pbar = tqdm(total=vid_frames)
while success:
    if count % 16 == 0:
        resized = cv2.resize(image, None, fx=SCALE, fy=SCALE)
        cv2.imwrite(OUTLOC + "%d.jpg" % count, resized)     # save frame as JPEG file
    success, image = vidcap.read()
    count += 1
    pbar.update(1)

pbar.close()