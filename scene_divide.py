import sys
import io
from PyQt5.QtCore import QThread, pyqtSignal
import contextlib

from transnetv2 import TransnetDetector
from scenedetect import detect, AdaptiveDetector


class StreamInterceptor(io.StringIO):
    """stdout을 실시간으로 가져오기 위한 클래스"""
    def __init__(self, signal : pyqtSignal):
        super().__init__()
        self.signal = signal

    def write(self, text : str):
        super().write(text)
        if text.strip(): # 빈 줄 제외
            self.signal.emit(text.strip())

    def flush(self):
        """Ignore flush"""
        pass


class SceneDivideThread(QThread):
    stdout_signal = pyqtSignal(str) # stdout
    result_signal = pyqtSignal(list) # scene list

    def __init__(self, video_path, start_frame=0, detect_model="scenedetect"):
        super().__init__()
        self.video_path = video_path
        self.detect_model = detect_model
        if detect_model == "scenedetect":
            self.scene_detector = AdaptiveDetector()
        elif detect_model == "transnetv2":
            self.scene_detector = TransnetDetector('transnetv2/transnetv2-pytorch-weights.pth', device='cpu')
        else:
            raise ValueError(f"detect model type {detect_model} is not supported")
        self.start_frame = start_frame

    def run(self):

        scene_list = []
        try:
            if self.detect_model == "scenedetect":
                split_list = detect(self.video_path, self.scene_detector, show_progress=True)
                for split in split_list:
                    scene = [split[0].get_frames(), split[1].get_frames() -1]
                    scene_list.append(scene)
            elif self.detect_model == "transnetv2":
                _, predictions, _ = self.scene_detector.predict_video(self.video_path,
                                                                    start_frame=self.start_frame)
                scene_list = self.scene_detector.predictions_to_scenes(predictions)
        finally:
            self.result_signal.emit(scene_list)