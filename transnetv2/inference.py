import os
import numpy as np
import torch
import argparse
from PIL import Image, ImageDraw

# ffmpeg-python 패키지가 필요합니다.
try:
    import ffmpeg
except ImportError:
    ffmpeg = None

from .transnetv2 import TransNetV2

def build_model(model_path, device="cpu"):
    model = TransNetV2()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    
    return model

class TransnetDetector:
    def __init__(self, model_path, device="cpu"):
        self._input_size = (27, 48, 3)

        self._model = build_model(model_path, device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)

    def predict_raw(self, frames: np.ndarray):
        """
        입력:
          frames: numpy array, shape = [batch, frames, height, width, 3]
        동작:
          numpy array를 torch tensor로 변환한 후 채널 순서를 [batch, frames, 3, height, width]로 바꿉니다.
          모델에 입력하여 두 종류의 예측(logits, many_hot)을 구하고 sigmoid를 적용한 결과를 반환합니다.
        """
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        # numpy -> tensor, float형 변환
        input_tensor = torch.from_numpy(frames).to(self.device)  # [B, T, H, W, 3] : model based in tensorflow

        with torch.no_grad():
            # 모델의 forward 함수가 (logits, dict)를 리턴한다고 가정합니다.
            logits, outputs = self._model(input_tensor)
            # 단일 프레임 예측 (예: shape [B, T, 1])에 sigmoid 적용
            single_frame_pred = torch.sigmoid(logits)
            # 만약 outputs가 dict라면 "many_hot" 키를 사용하고, 아니라면 바로 outputs를 사용합니다.
            many_hot = outputs["many_hot"] if isinstance(outputs, dict) and "many_hot" in outputs else outputs
            all_frames_pred = torch.sigmoid(many_hot)

        return single_frame_pred.cpu().numpy(), all_frames_pred.cpu().numpy()

    def predict_frames(self, frames: np.ndarray):
        """
        입력:
          frames: numpy array, shape = [frames, height, width, 3]
        동작:
          프레임에 대해 슬라이딩 윈도우(길이 100, step=50) 단위로 패딩 후 예측을 수행합니다.
          예측 결과 중 중간 50 프레임(인덱스 25~75)을 추출하여 최종 결과를 만듭니다.
        """
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            no_padded_frames_start = 25
            remainder = len(frames) % 50
            extra = 50 - remainder if remainder != 0 else 50
            no_padded_frames_end = 25 + extra  # 마지막 패딩 프레임 수

            # 시작/끝 프레임을 복사하여 패딩합니다.
            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [np.repeat(start_frame, no_padded_frames_start, axis=0),
                 frames,
                 np.repeat(end_frame, no_padded_frames_end, axis=0)],
                axis=0
            )
            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]  # 배치 차원 추가

        predictions = []
        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            # inp의 shape: [1, 100, H, W, 3] 에서 중간 50 프레임(인덱스 25~75)을 추출
            predictions.append((single_frame_pred[0, 25:75, 0], all_frames_pred[0, 25:75, 0]))
            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred_all = np.concatenate([p[0] for p in predictions], axis=0)
        all_frames_pred_all = np.concatenate([p[1] for p in predictions], axis=0)
        # 패딩된 초과 부분 제거
        return single_frame_pred_all[:len(frames)], all_frames_pred_all[:len(frames)]

    def predict_video(self, video_fn: str, start_frame: int = 0):
        """
        입력:
          video_fn: 동영상 파일 경로.
        동작:
          ffmpeg를 이용하여 48x27 크기의 RGB 프레임을 추출하고,
          predict_frames()를 호출하여 예측합니다.
        출력:
          (video_frames, single_frame_predictions, all_frame_predictions)
        """
        if ffmpeg is None:
            raise ModuleNotFoundError(
                "For `predict_video` function `ffmpeg` needs to be installed. "
                "Install `ffmpeg` command line tool and then run `pip install ffmpeg-python`."
            )

        print("[TransNetV2] Extracting frames from {}".format(video_fn))

        probe = ffmpeg.probe(video_fn, v='error', select_streams='v:0', show_entries='stream=r_frame_rate')
        # r_frame_rate 값은 'numerator/denominator' 형태이므로 이를 분리하여 계산
        rate = probe['streams'][0]['r_frame_rate']
        numerator, denominator = map(int, rate.split('/'))
        frame_rate = numerator / denominator
        start_time = start_frame / frame_rate

        out, err = (
            ffmpeg
            .input(video_fn, ss=start_time)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27", vsync="passthrough")
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, 27, 48, 3])
        video_frames = video
        single_frame_predictions, all_frame_predictions = self.predict_frames(video)
        return video_frames, single_frame_predictions, all_frame_predictions

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        """
        입력:
          predictions: 단일 프레임 예측 (numpy array)
          threshold: 이진화 기준
        동작:
          예측값을 이진화한 후 연속된 장면(scene)을 [start, end] 형태의 numpy array로 반환합니다.
        """
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t_prev = 0
        start = 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if predictions[-1] == 0:
            scenes.append([start, len(predictions) - 1])
        if len(scenes) == 0:
            return [[0, len(predictions) - 1]]
        return scenes

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        """
        입력:
          frames: numpy array, shape = [frames, height, width, 3]
          predictions: 단일 프레임 예측 및 전체 프레임 예측 (튜플 또는 list)
        동작:
          영상 프레임을 그리드 이미지로 재배치하고 예측값을 선으로 그려 시각화합니다.
        """
        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25
        pad_with = width - (len(frames) % width) if (len(frames) % width) != 0 else 0
        frames_padded = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)], mode='edge')
        predictions_padded = [np.pad(x, (0, pad_with), mode='edge') for x in predictions]
        height = len(frames_padded) // width

        # reshape을 통해 그리드 형태로 만듭니다.
        img = frames_padded.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(np.split(np.concatenate(np.split(img, height), axis=2)[0], width), axis=2)[0, :-1]
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for i, pred_vals in enumerate(zip(*predictions_padded)):
            x, y = i % width, i // width
            x = x * (iw + len(predictions)) + iw
            y = y * (ih + 1) + ih - 1

            for j, p in enumerate(pred_vals):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255
                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)
        return img