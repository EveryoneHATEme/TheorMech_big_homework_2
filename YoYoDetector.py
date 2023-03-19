import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from typing import Tuple


class YoYoDetector:
    class CouldNotOpenFile(Exception):
        pass

    class ThresholdError(Exception):
        pass

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.capture = cv.VideoCapture(self.video_path)
        self.fps = self.capture.get(cv.CAP_PROP_FPS) * 4  # the original framerate was 120 fps
        self.frame_count = self.capture.get(cv.CAP_PROP_FRAME_COUNT)
        self.video_duration = self.frame_count / self.fps

        if not self.capture.isOpened():
            raise YoYoDetector.CouldNotOpenFile(f"Couldn't open the file {self.video_path}")

    def get_video_length(self):
        framerate = self.capture.get(cv.CAP_PROP_FPS)
        frame_count = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))
        return frame_count // framerate

    @staticmethod
    def to_grayscale(image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    @staticmethod
    def make_threshold(image):
        ok, threshold_image = cv.threshold(image, 50, 255, cv.THRESH_BINARY_INV)
        if not ok:
            raise YoYoDetector.ThresholdError()
        return threshold_image

    @staticmethod
    def get_contours(image):
        contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        return contours, hierarchy

    @staticmethod
    def cut_image(image):
        return image[500:-200, 20:]

    @staticmethod
    def process_image(image):
        image = YoYoDetector.cut_image(image)
        gray_image = YoYoDetector.to_grayscale(image)
        threshold_image = YoYoDetector.make_threshold(gray_image)
        return threshold_image

    @staticmethod
    def get_yoyo_rectangle(contours):
        if len(contours) > 1:
            contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)
            first_rectangle, second_rectangle = list(map(cv.boundingRect, contours_sorted))[:2]

            # if second rectangle is inside first, then first rectangle is yoyo and second is the white dot on yoyo
            # if it is not then rectangle with highest y value is yoyo and another one is the hand
            if Utils.is_inside(first_rectangle, second_rectangle):
                yoyo_rectangle = first_rectangle
            else:
                yoyo_rectangle = max((first_rectangle, second_rectangle), key=itemgetter(1))
        elif len(contours) == 1:
            yoyo_rectangle = cv.boundingRect(contours[0])
        else:
            yoyo_rectangle = None

        return yoyo_rectangle

    def play_processed_video(self):
        while self.capture.isOpened():
            ok, image = self.capture.read()
            if not ok:
                break

            processed_image = self.process_image(image)
            contours, hierarchy = self.get_contours(processed_image)
            yoyo_rectangle = self.get_yoyo_rectangle(contours)
            if yoyo_rectangle is None:
                continue
            x, y, w, h = yoyo_rectangle
            image = self.cut_image(image)
            image = cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0, 1))
            # cv.drawContours(image, contours, -1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

            cv.imshow(self.video_path, image)

            key = cv.waitKey(20)
            if key == ord('q'):
                break

    def get_y_history(self):
        y_history = []
        while self.capture.isOpened():
            ok, image = self.capture.read()
            if not ok:
                break

            processed_image = self.process_image(image)
            contours, hierarchy = self.get_contours(processed_image)
            yoyo_rectangle = self.get_yoyo_rectangle(contours)
            if yoyo_rectangle is None:
                continue
            x, y, w, h = yoyo_rectangle
            center_y = y + h // 2

            y_history.append(center_y)
        return np.array(y_history)

    def get_timeline(self, y_history):
        return np.linspace(0, self.video_duration, len(y_history))

    def __del__(self):
        self.capture.release()
        cv.destroyAllWindows()


class Plotter:
    def __init__(self, rope_length: float, y_history: list[np.ndarray],
                 timeline: list[np.ndarray], experiment_number: int = None):
        self.rope_length = rope_length
        self.y_history = [np.array(arr, dtype=np.float64) for arr in y_history]
        self.timeline = timeline
        self.experiment_number = experiment_number

    def plot_all_in_one(self):
        pixels_per_cm = [(np.max(arr) - np.min(arr)) / self.rope_length for arr in self.y_history]
        for history, pixels in zip(self.y_history, pixels_per_cm):
            history -= np.min(history)
            history /= pixels
        figure, axes = plt.subplots()

        if not os.path.exists('plots'):
            os.mkdir('plots')

        for i, (history, timeline) in enumerate(zip(self.y_history, self.timeline), start=1):
            axes.plot(timeline, history, label=f'{i} try')
        axes.set_xlabel('time, s')
        axes.set_ylabel('length, cm')
        axes.axhline(self.rope_length, label='initial length', linestyle='--')
        axes.legend()
        axes.set_yticks(np.linspace(0, self.rope_length, 20))
        axes.grid()
        if self.experiment_number:
            axes.set_title(f'Experiment number {self.experiment_number} with length {self.rope_length} cm')
            plt.savefig(f'plots/ex{self.experiment_number}')
        plt.show()

    def plot_one_by_one(self):
        pixels_per_cm = [(np.max(arr) - np.min(arr)) / self.rope_length for arr in self.y_history]
        for history, pixels in zip(self.y_history, pixels_per_cm):
            history -= np.min(history)
            history /= pixels

        if not os.path.exists('plots'):
            os.mkdir('plots')
        if not os.path.exists(f'plots/ex{self.experiment_number}'):
            os.mkdir(f'plots/ex{self.experiment_number}')

        for i, (history, timeline) in enumerate(zip(self.y_history, self.timeline), start=1):
            figure, axes = plt.subplots()
            axes.plot(timeline, history, label=f'{i} try')
            axes.set_xlabel('time, s')
            axes.set_ylabel('length, cm')
            axes.axhline(self.rope_length, label='initial length', linestyle='--')
            axes.legend()
            axes.set_yticks(np.linspace(0, self.rope_length, 20))
            axes.grid()
            if self.experiment_number:
                axes.set_title(f'Experiment number {self.experiment_number} with length {self.rope_length} cm')
                plt.savefig(f'plots/ex{self.experiment_number}/{i}')

    def plot_average(self):
        pixels_per_cm = [(np.max(arr) - np.min(arr)) / self.rope_length for arr in self.y_history]
        for history, pixels in zip(self.y_history, pixels_per_cm):
            history -= np.min(history)
            history /= pixels
        figure, axes = plt.subplots()

        if not os.path.exists('plots'):
            os.mkdir('plots')

        average = []

        for history in zip(*self.y_history):
            average.append(sum(history) / len(history))

        axes.plot(min(self.timeline, key=len), average, label=f'average')
        axes.set_xlabel('time, s')
        axes.set_ylabel('length, cm')
        axes.axhline(self.rope_length, label='initial length', linestyle='--')
        axes.legend()
        axes.set_yticks(np.linspace(0, self.rope_length, 20))
        axes.grid()
        if self.experiment_number:
            axes.set_title(f'Experiment number {self.experiment_number} with length {self.rope_length} cm')
            plt.savefig(f'plots/ex{self.experiment_number}')
        plt.show()


class Utils:
    @staticmethod
    def is_inside(first_rectangle: Tuple[int, int, int, int], second_rectangle: Tuple[int, int, int, int]) -> bool:
        """
        checks if first_rectangle contains second_rectangle
        """
        x1, y1, w1, h1 = first_rectangle
        x2, y2, w2, h2 = second_rectangle

        horizontal = x1 <= x2 and x1 + w1 >= x2 + w2
        vertical = y1 <= y2 and y1 + h1 >= y2 + h2

        return horizontal and vertical


if __name__ == '__main__':
    rope_lengths = {
        '1': 63,
        '2': 53,
        '3': 43,
        '4': 33,
        '5': 23
    }

    histories = dict()
    timelines = dict()

    files = os.listdir('video')
    try:
        for filename in files:
            experiment_index = filename[:filename.find('.')].split('_')
            detector = YoYoDetector(os.path.join('video', filename))
            current_history = detector.get_y_history()
            current_timeline = detector.get_timeline(current_history)
            if experiment_index[0] in histories:
                histories[experiment_index[0]].append(current_history)
                timelines[experiment_index[0]].append(current_timeline)
            else:
                histories[experiment_index[0]] = [current_history]
                timelines[experiment_index[0]] = [current_timeline]
    except KeyboardInterrupt:
        print('interrupted, finishing')

    for key, value in histories.items():
        plotter = Plotter(rope_lengths[key], histories[key], timelines[key], experiment_number=int(key))
        plotter.plot_average()
