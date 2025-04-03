import cv2
import numpy as np
import multiprocessing as mp
from typing import List
from enum import IntEnum
from PyQt5.QtCore import QThread, pyqtSignal
from utils import Option
import os
import tempfile
import subprocess

INTER_MODE = {
    'bicubic': cv2.INTER_CUBIC,
    'bilinear': cv2.INTER_LINEAR,
    'lanczos': cv2.INTER_LANCZOS4,
    'area': cv2.INTER_AREA,
}

class FilterType(IntEnum):
    DOWN_AND_UP = 0
    NL_DENOISE = 1
    TCOMB = 2


def apply_filter(image: np.array, filter_type: FilterType, filter_opt: Option):
    """
    Currently, only support Up & Down filtering

    args:
        image:
        filter_type:
        filter_opts:
    """
    if filter_type == FilterType.DOWN_AND_UP:
        option = filter_opt.down_and_up
        mode = INTER_MODE[option.mode]
        N = option.scale

        down_image = cv2.resize(image, None, fx=1/N, fy=1/N, interpolation=mode)
        up_image = cv2.resize(down_image, (image.shape[1], image.shape[0]), interpolation=mode)

        return up_image
    elif filter_type == FilterType.NL_DENOISE:
        option = filter_opt.nl_denoise
        filter_h = option.filter_h

        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            filter_h,
            filter_h,
            7,
            21,
        )
        return denoised
    elif filter_type == FilterType.TCOMB:
        option = filter_opt.tcomb
        # TComb는 앞뒤 프레임 정보를 필요로 하므로 단일 프레임 처리 불가
        # 이 함수는 단일 이미지 처리만 가능하므로 원본 반환
        return image
    else:
        raise ValueError(f"Filter type {filter_type.name} is not Supported")

def get_mask(image: np.array, denoised: np.array):
    mask = cv2.cvtColor(cv2.absdiff(image, denoised), cv2.COLOR_RGB2GRAY)
    return mask


class TCombFilterThread(QThread):
    """TComb 필터를 비디오 전체에 적용하는 스레드"""
    progress_signal = pyqtSignal(int)  # 진행률 (%)
    result_signal = pyqtSignal()

    def __init__(self,
                 video_path,
                 filter_opts: Option,
                 origin_images: np.memmap,
                 filtered_images: np.memmap,
                 mask_images: np.memmap,
                 score_list: np.array,
                 is_loaded: np.array,
                 start_frame: int = 0,
                 end_frame: int = -1):  # -1은 영상 끝까지 처리
        super().__init__()
        self.video_path = video_path
        self.filter_opts = filter_opts
        self.origin_images = origin_images
        self.filtered_images = filtered_images
        self.mask_images = mask_images
        self.score_list = score_list
        self.is_loaded = is_loaded
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.running = True
        self.temp_dir = tempfile.gettempdir()
        self.process = None
        self.orig_cap = None
        self.filtered_cap = None

    def run(self):
        if not self.running:
            return

        option = self.filter_opts.tcomb
        
        # 1. 비디오 정보 가져오기
        cap = cv2.VideoCapture(self.video_path)
        frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 캡처 객체 닫기
        cap.release()
        
        # 종료 프레임이 -1이면 영상 끝까지 처리
        if self.end_frame < 0:
            self.end_frame = frame_len - 1
        
        # 프레임 범위 유효성 검사
        self.start_frame = max(0, min(self.start_frame, frame_len - 1))
        self.end_frame = max(self.start_frame, min(self.end_frame, frame_len - 1))
        
        print(f"TComb 처리 범위: {self.start_frame}~{self.end_frame} (총 {self.end_frame - self.start_frame + 1}프레임)")
        
        # 임시 파일 경로
        temp_script = os.path.join(self.temp_dir, f"tcomb_script_{self.start_frame}_{self.end_frame}.py")
        temp_y4m = os.path.join(self.temp_dir, f"tcomb_result_{self.start_frame}_{self.end_frame}.y4m")
        
        try:
            # 2. 임시 스크립트 파일 생성
            # 패딩 계산
            pad_right = 1 if (W % 2 != 0) else 0
            pad_bottom = (4 - (H % 4)) if (H % 4 != 0) else 0
            
            with open(temp_script, "w") as f:
                f.write(f"""import vapoursynth as vs

core = vs.core
core.num_threads = 10

mode = {option.mode}      # 0: luma, 1: chroma, 2: luma+chroma
fthreshl = {option.fthreshl}  # 밝기 프레임 내 변화 감지 임계값
fthreshc = {option.fthreshc}  # 색상 프레임 내 변화 감지 임계값
othreshl = {option.othreshl}  # 밝기 프레임 간 변화 감지 임계값
othreshc = {option.othreshc}  # 색상 프레임 간 변화 감지 임계값
scthresh = {option.scthresh}

# 원본 비디오 불러오기
src = core.ffms2.Source(source="{self.video_path}")

# 원하는 프레임 범위만 선택
src = src[{self.start_frame}:{self.end_frame+1}]

W = src.width
H = src.height

# 패딩 적용
pad_right = {pad_right}
pad_bottom = {pad_bottom}

if pad_right > 0 or pad_bottom > 0:
    src = core.std.AddBorders(
        src,
        left=0,
        right=pad_right,
        top=0,
        bottom=pad_bottom,
        color=[0, 0, 0]
    )

down8 = core.resize.Bicubic(src, format=vs.YUV420P8)

tc_processed_8 = core.tcomb.TComb(
    down8, 
    mode=mode, 
    fthreshl=fthreshl,
    fthreshc=fthreshc,
    othreshl=othreshl,
    othreshc=othreshc,
    map=0,
    scthresh=scthresh
)
tc_mask_8 = core.tcomb.TComb(
    down8, 
    mode=mode,
    fthreshl=fthreshl,
    fthreshc=fthreshc,
    othreshl=othreshl,
    othreshc=othreshc,
    map=1,
    scthresh=scthresh
)

tc_processed = core.resize.Bicubic(tc_processed_8, format=src.format.id)
tc_mask = core.resize.Bicubic(tc_mask_8, format=src.format.id)

# 원본과 필터링 결과를 마스크를 이용해 합성
dst = core.std.MaskedMerge(src, tc_processed, tc_mask)

# 패딩 제거 - 필터링된 영상에만 적용
if pad_right > 0 or pad_bottom > 0:
    dst = core.std.Crop(dst, right=pad_right, bottom=pad_bottom)

dst.set_output()
""")

            # 3. VapourSynth 실행
            vspipe_cmd = [
                "vspipe",
                "-c", "y4m",           # y4m 형식으로 출력
                "-p",                  # 진행률 표시
                temp_script,           # 스크립트 파일
                temp_y4m               # 출력 파일
            ]
            
            # 필터링된 영상 처리 - 리소스 관리를 위해 컨텍스트 매니저 사용
            print(f"실행: {' '.join(vspipe_cmd)}")
            
            # 파이프를 최소한으로 유지
            self.process = subprocess.Popen(
                vspipe_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                close_fds=True  # 모든 추가 파일 디스크립터 닫기
            )
            _, stderr = self.process.communicate()
            if self.process.returncode != 0:
                error_msg = stderr.decode()
                print(f"VapourSynth error: {error_msg}")
                self.result_signal.emit()
                return
                
            # 4. 결과 영상 가져오기
            # 원본 동영상 및 결과 파일 열기
            self.orig_cap = cv2.VideoCapture(self.video_path)
            self.orig_cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)  # 시작 프레임으로 이동
            
            self.filtered_cap = cv2.VideoCapture(temp_y4m)
            
            # 원본 및 필터링된 프레임 해상도 확인
            _, test_orig_frame = self.orig_cap.read()
            orig_h, orig_w = test_orig_frame.shape[:2]
            self.orig_cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)  # 다시 시작 프레임으로 돌아감
            
            # 결과 해상도 확인
            _, test_filtered_frame = self.filtered_cap.read()
            filtered_h, filtered_w = test_filtered_frame.shape[:2]
            self.filtered_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 처리된 결과의 첫 프레임으로 돌아감
            
            print(f"원본 해상도: {orig_w}x{orig_h}, 필터링 해상도: {filtered_w}x{filtered_h}")
            
            if (orig_h != filtered_h or orig_w != filtered_w):
                print(f"경고: 해상도 불일치! 원본({orig_w}x{orig_h}), 필터링({filtered_w}x{filtered_h})")
            
            # 현재 프레임 인덱스
            frame_count = 0
            total_frames = self.end_frame - self.start_frame + 1
            
            # 현재 씬에 속한 프레임만 처리
            for i in range(self.start_frame, self.end_frame + 1):
                if not self.running:
                    break
                    
                frame_count += 1
                    
                # 진행률 업데이트
                if frame_count % 5 == 0 or frame_count == total_frames:
                    progress = int((frame_count / total_frames) * 100)
                    self.progress_signal.emit(progress)
                    
                # 원본 프레임 읽기
                ret, original_frame = self.orig_cap.read()
                if not ret:
                    break
                    
                # 필터링된 프레임 읽기
                ret_filtered, filtered_frame = self.filtered_cap.read()
                if not ret_filtered:
                    break
                    
                # BGR -> RGB 변환
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
                
                # 해상도가 다른 경우 원본 크기에 맞춤
                if filtered_frame.shape[:2] != (orig_h, orig_w):
                    filtered_frame = cv2.resize(filtered_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
                mask = get_mask(original_frame, filtered_frame)
                # _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
                    
                # 데이터 저장
                self.origin_images[i] = original_frame
                self.filtered_images[i] = filtered_frame
                self.mask_images[i] = mask
                self.score_list[i] = mask.mean()
                self.is_loaded[i] = True
        
        except Exception as e:
            print(f"TComb 처리 중 오류 발생: {e}")
        finally:
            # 모든 리소스 정리
            self.cleanup_resources()
            
            # 작업 완료 신호 보내기
            self.progress_signal.emit(100)
            self.result_signal.emit()
            
    def cleanup_resources(self):
        """리소스 정리 메서드"""
        # 프로세스 정리
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except (subprocess.TimeoutExpired, AttributeError):
                try:
                    self.process.kill()
                except:
                    pass
                    
        # 캡처 객체 정리
        if self.orig_cap is not None:
            self.orig_cap.release()
            self.orig_cap = None
            
        if self.filtered_cap is not None:
            self.filtered_cap.release()
            self.filtered_cap = None
            
        # 임시 파일 삭제
        try:
            temp_script = os.path.join(self.temp_dir, f"tcomb_script_{self.start_frame}_{self.end_frame}.py")
            temp_y4m = os.path.join(self.temp_dir, f"tcomb_result_{self.start_frame}_{self.end_frame}.y4m")
            
            if os.path.exists(temp_script):
                os.remove(temp_script)
            if os.path.exists(temp_y4m):
                os.remove(temp_y4m)
        except Exception as e:
            print(f"임시 파일 삭제 중 오류: {e}")
        
    def stop(self):
        """스레드 중지 메서드"""
        self.running = False
        # 리소스 정리 호출
        self.cleanup_resources()


class VideoLoadAndFilterThread(QThread):
    """비디오를 순차적으로 읽고 프레임 단위로 필터링 적용하는 클래스"""
    frame_signal = pyqtSignal(int) # index
    result_signal = pyqtSignal()

    def __init__(self,
                 video_path,
                 filter_type: FilterType,
                 filter_opts: Option,
                 origin_images: np.memmap,
                 filtered_images: np.memmap,
                 mask_images: np.memmap,
                 score_list: np.array,
                 is_loaded: np.array,
                 start_index: int = 0,
                 end_index: int = -1,
                 chunk_size: int = 100,
                 buffer_size: int = 100,
                 ):
        super().__init__()
        self.video_path = video_path
        self.filter_type = filter_type
        self.filter_opts = filter_opts
        self.origin_images = origin_images
        self.filtered_images = filtered_images
        self.mask_images = mask_images

        self.score_list = score_list # Shared variable
        self.is_loaded = is_loaded # Shared variable

        self.start_index = start_index
        self.end_index = end_index
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size # Request에서 같이 읽어올 인접 프레임 개수
        self.running = True
        self.cap = None
        self.tcomb_thread = None

        # 멀티 프로세싱 큐 및 상태 관리
        self.request_queue = mp.Queue()

    def request_frame(self, frame_idx):
        self.request_queue.put(frame_idx)

    def run(self):
        try:
            # TComb 필터는 별도의 방식으로 처리
            if self.filter_type == FilterType.TCOMB:
                self.tcomb_thread = TCombFilterThread(
                    self.video_path,
                    self.filter_opts,
                    self.origin_images,
                    self.filtered_images,
                    self.mask_images,
                    self.score_list,
                    self.is_loaded,
                    start_frame=self.start_index,
                    end_frame=self.end_index
                )
                # 시그널 연결
                self.tcomb_thread.progress_signal.connect(lambda progress: self.frame_signal.emit(
                    int(self.start_index + ((self.end_index - self.start_index) * progress / 100))
                ))
                self.tcomb_thread.result_signal.connect(self.result_signal)
                
                # 스레드 실행
                self.tcomb_thread.start()
                
                # moveToThread 대신 wait() 사용 - 리소스 관리 개선
                self.tcomb_thread.wait()
                return
                
            self.cap = cv2.VideoCapture(self.video_path)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_index)
            frame_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.end_index < 0:
                self.end_index = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
            self.running = True
            index = self.start_index
            chunk_count = 0
            while self.running:
                if index >= self.end_index:
                    break
                if self.is_loaded[index]:
                    index += 1
                    continue
    
                # 순차적 로드
                ret, frame = self.cap.read()
                if not ret:
                    break
    
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                filtered_frame = apply_filter(original_frame, self.filter_type, self.filter_opts)
                mask = get_mask(original_frame, filtered_frame)
                mask_mean = mask.mean()
    
                self.origin_images[index] = original_frame
                self.filtered_images[index] = filtered_frame
                self.mask_images[index] = mask
                self.score_list[index] = mask_mean
                self.is_loaded[index] = True
    
                index += 1
                chunk_count += 1
    
                if chunk_count > self.chunk_size:
                    self.frame_signal.emit(index)
                    chunk_count = 0
    
            if chunk_count > 0:
                self.frame_signal.emit(index)
        
        except Exception as e:
            print(f"비디오 로딩 중 오류 발생: {e}")
        finally:
            self.cleanup_resources()
            self.result_signal.emit()
            self.running = False

    def cleanup_resources(self):
        """리소스 정리 메서드"""
        # 캡처 객체 정리
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # TComb 스레드 정리
        if self.tcomb_thread is not None and self.tcomb_thread.isRunning():
            self.tcomb_thread.stop()
            self.tcomb_thread.wait(1000)
            self.tcomb_thread = None
            
        # 큐 정리
        try:
            # 큐 비우기
            while not self.request_queue.empty():
                self.request_queue.get_nowait()
        except:
            pass

    def stop(self):
        """스레드 중지 메서드"""
        self.running = False
        
        # TComb 스레드가 실행 중이면 중지
        if self.tcomb_thread is not None and self.tcomb_thread.isRunning():
            self.tcomb_thread.stop()
            
        # 리소스 정리 호출
        self.cleanup_resources()