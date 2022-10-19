# -*- coding: utf-8 -*- 
import subprocess
import os

if __name__ == "__main__":
    # devnull = open('/dev/null', 'w')
    devnull = open(os.devnull, 'w')  # /dev/null へ吐き出すことでコンソール出力させない。
    vis_proc = subprocess.Popen(
        ['python', 'vis_server.py'], stdout=devnull, stderr=devnull)

    reid_proc = subprocess.Popen(
        ['python', 'face_reid_server.py'], stdout=devnull, stderr=devnull)

    face_det_proc = subprocess.Popen(
        ['python', 'face_det_server.py'], stdout=devnull, stderr=devnull)

    cam_proc = subprocess.Popen(
        ['python', 'cam_server.py'], stdout=devnull, stderr=devnull)

    print("Start visualizer.")

    while True:
        print("Enter <q>, if terminate visualizer.")
        key_val = input()
        if key_val == "q":
            vis_proc.kill()
            reid_proc.kill()
            face_det_proc.kill()
            cam_proc.kill()
            break
    print("Terminate visualizer.")
