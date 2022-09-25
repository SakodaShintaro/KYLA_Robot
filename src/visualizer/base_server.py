# -*- coding: utf-8 -*-
import queue
import socket
import json

class BaseServer(object):
    def __init__(self):
        # self.frame_data = queue.Queue(maxsize=1)
        self.frame_data = queue.Queue()

    def update(self):
        pass

    def put_frame_data(self, frame_data):
        self.frame_data.put(frame_data)

    def execute(self):
        while True:
            self.update()
