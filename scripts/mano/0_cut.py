import glob
import os.path as osp
import threading
import time
import traceback
from pathlib import Path
from pprint import pprint
import sys
xgym_path = Path(__file__).resolve().parents[2]  # Adjust path to ~/repos/xgym
sys.path.append(str(xgym_path))
import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ImageEnhance
from pynput import keyboard
from pynput.keyboard import Key
from scipy.optimize import least_squares
from tqdm import tqdm
from xgym.controllers import KeyboardController
from xgym.utils import camera as cu
from xgym.__init__ import MANO_0, MANO_1
# import bala


def load_images(data_dir):
    images = [Image.open(p) for p in Path(data_dir).rglob("*.jpg")]
    return images


def tryex(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            print(e)
            return None

    return wrapper


class VideoCutter:
    def __init__(self, data_dir: Path = MANO_0):
        self.data_dir = data_dir
        self.out_dir = MANO_1
        self.videos = list(data_dir.glob("*.mp4"))

        # self.reader = imageio.get_reader(video_path)
        # self.frames = {i: x for i, x in enumerate(self.reader)}

        self.lock = threading.Lock()

        kb = KeyboardController()
        kb.register("q", self.quit)
        kb.register("c", self.cut)
        kb.register("d", self.delete)
        kb.register("s", self.save)
        kb.register("r", self.reset)
        kb.register(Key.space, self.play)
        # kb.register(Key.enter, lambda: self.proceed())

        kb.register(Key.up, self.previous_video)
        kb.register(Key.down, self.next_video)
        kb.register(Key.left, self.decrement_frame)
        kb.register(Key.right, self.increment_frame)
        kb.register(Key.shift, self.step_size)
        self.kb = kb

        self.nvideo = 0
        self.reset()
        self.start()

    def reset(self):
        self.c1, self.c2 = None, None
        self.nframe = 0
        self.prev = -1
        self.step = 1
        self.saved = False
        self._play = False
        self._running = True

        self.reader = imageio.get_reader(self.videos[self.nvideo])
        self.frames = {0: self.reader.get_data(0)}

        print(f"Resetting to video {self.videos[self.nvideo]}")
        print(self.nframe)
        print(self.frames)

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            print("No OpenCV windows to destroy.")

    def next_video(self):
        self.reset()
        self.nvideo += 1
        self.nvideo %= len(self.videos) # loop around
        print(self.videos[self.nvideo])

    def previous_video(self):
        self.reset()
        self.nvideo -= 1
        self.nvideo %= len(self.videos)
        print(self.videos[self.nvideo])

    def quit(self):
        with self.lock:
            print("Quitting...")
            self._running = False
            self.kb.close()
            self.display = self.quit
            print("Closed keyboard")
            time.sleep(0.1)
            done_dir = self.data_dir.parent / "0_input_done"
            done_dir.mkdir(exist_ok=True)

            # Move all processed videos to done_dir
            for video in self.videos:
                dest_path = done_dir / video.name
                video.rename(dest_path)
                print(f"Moved {video} to {dest_path}")

        exit(0)

    def delete(self):
        self.frames[self.nframe] = None

        if self.c1 is not None and self.c2 is not None:
            c1, c2 = min(self.c1, self.c2), max(self.c1, self.c2)
            frames = {i: self[i] for i in range(c1, c2 + 1) if self[i] is not None}
            for i in frames.keys():
                self.frames[i] = None

            print({k: type(v) for k, v in self.frames.items()})

        self.increment_frame() if self.prev <= self.nframe else self.decrement_frame()

    @tryex
    def step_size(self):
        transition = {1: 10, 10: 100, 100: 1}
        self.step = transition[self.step]

    @tryex
    def increment_frame(self, n=None):
        if n == 0:
            return
        self.prev = self.nframe
        self.nframe += n if not n is None else self.step
        self.nframe = min(len(self.reader) - 1, self.nframe)
        print(self.prev, self.nframe)

    @tryex
    def decrement_frame(self):
        self.prev = self.nframe
        self.nframe = self.nframe - self.step
        self.nframe = max(0, self.nframe)
        print(self.prev, self.nframe)

    @tryex
    def next_video(self):
        self.nvideo += 1
        self.nvideo %= len(self.videos)
        self.reader = imageio.get_reader(self.videos[self.nvideo])
        self.frames = {i: x for i, x in enumerate(self.reader)}

    @tryex
    def previous_video(self):
        self.nvideo -= 1
        self.nvideo %= len(self.videos)
        self.reader = imageio.get_reader(self.videos[self.nvideo])
        self.frames = {i: x for i, x in enumerate(self.reader)}

    @tryex
    def cut(self):
        if self.c1 is None:
            self.c1 = self.nframe
        elif self.c2 is None:
            self.c2 = self.nframe
        else:
            self.c1, self.c2 = None, None

    def __getitem__(self, key):
        with self.lock:
            frame = self.frames.get(key, np.asarray(self.reader.get_data(key)))
            self.frames[key] = frame
            return frame

    @tryex
    def save(self):
        if self.c1 is None or self.c2 is None:
            print("No frames selected")
            return

        c1, c2 = min(self.c1, self.c2), max(self.c1, self.c2)
        frames = [self[i] for i in range(c1, c2 + 1) if self[i] is not None]

        frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]

        video = self.videos[self.nvideo]
        fname = self.out_dir / f"{video.stem}_{c1}_{c2}.mp4"
        fname = str(fname)

        cu.save_frames(frames, fname, ext="mp4", fps=30)

        print(f"Saved {fname}")
        self.saved = True
        self.c1, self.c2 = None, None

    def play(self):
        self._play = not self._play

    def span(self):
        val = self[self.nframe]
        if val is None:
            print(f"spanning {self.nframe}")
            self.increment_frame() if self.prev < self.nframe or self.nframe <= 0 else self.decrement_frame()
            return self.span()
        return val

    @tryex
    def display(self):
        print(f"frame: {self.nframe}")
        self.frame = self.span()

        color = (100, 255, 100) if self.saved else (255, 100, 255)
        frame = cv2.resize(self.frame.copy(), (800, 800))
        frame = cv2.putText(
            frame,
            f"frame:{self.nframe}, video:{self.nvideo}, step:{self.step} c1: {self.c1}, c2: {self.c2}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

        frame = cv2.putText(
            frame,
            "Saved" if self.saved else "Not saved",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(f'{self.videos[self.nvideo]}', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        self.saved = False
        wait = int(self._play)
        cv2.waitKey(wait)
        self.increment_frame(wait)

    def start(self):

        while self._running:
            self.display()


def main():
    v = VideoCutter()


if __name__ == "__main__":
    main()
