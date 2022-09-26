# -*- coding: utf-8 -*- 
import subprocess
import time
import os

if __name__ == "__main__":
    # devnull = open('/dev/null', 'w')
    devnull = open(os.devnull, 'w')  # /dev/null へ吐き出すことでコンソール出力させない。
    vis_proc = subprocess.Popen(
        ['python', 'vis_server.py'], stdout=devnull, stderr=devnull)
    time.sleep(3)

    face_det_proc = subprocess.Popen(
        ['python', 'face_det_server.py'], stdout=devnull, stderr=devnull)
    time.sleep(3)

    cam_proc = subprocess.Popen(
        ['python', 'cam_server.py'], stdout=devnull, stderr=devnull)

    print("Start visualizer.")

    while True:
        print("Enter <q>, if terminate visualizer.")
        key_val = input()
        if key_val == "q":
            vis_proc.kill()
            face_det_proc.kill()
            cam_proc.kill()
            break
    print("Terminate visualizer.")
