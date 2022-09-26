# -*- coding: utf-8 -*- 
# ref: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import time
import numpy as np

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 64850))
connection = client_socket.makefile('wb')

# video_file = "./face_mesh_input.mp4"
# cam = cv2.VideoCapture(video_file)
cam = cv2.VideoCapture(0)

# cam.set(3, 320);
# cam.set(4, 240);
fps = cam.get(cv2.CAP_PROP_FPS)
print(fps)

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        continue
    # frame = np.array(frame, dtype=np.uint8)
    result, frame = cv2.imencode('.jpg', frame, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    size_of_frame = frame.shape[0]

    print("{}: {}".format(img_counter, size_of_frame))
    # client_socket.sendall(struct.pack(">L", size_of_data) + data)

    # 決まったサイズでヘッダーをつけて、受け取り側でペイロードの大きさが分かるようにする。
    constant_sized_header = struct.pack(">L", size_of_frame) 
    client_socket.sendall(constant_sized_header + frame.tostring())
    # client_socket.sendall(frame)
    img_counter += 1
    # time.sleep(1.0 / fps)

cam.release()