import os
import os.path as osp
import sys
import cv2
import numpy as np
import time
import yaml
import tempfile
import atexit
import threading
import subprocess
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QScrollArea, QTableWidget, QTableWidgetItem,
                             QPushButton, QFileDialog, QSplitter, QFormLayout, QSlider,
                             QToolButton, QSizePolicy, QComboBox, QAbstractItemView)
from widget_utils import KeyTracker
from utils import Option
from filter_video import FilterType, VideoLoadAndFilterThread, TCombFilterThread, apply_filter, get_mask
from scene_divide import SceneDivideThread

import multiprocessing as mp

TEMP_DIR = tempfile.gettempdir()


class SceneTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_ = parent
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Start", "End", "Mean diff"])

        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.setDropIndicatorShown(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setDragDropOverwriteMode(False)


class DividerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.visible_flag = True
        self.setMinimumSize(15, 30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        if not self.visible_flag:
            return

        painter = QPainter(self)
        painter.setBrush(QColor(255, 165, 0))
        painter.setPen(QColor(255, 100, 0))

        rect = QRect(0, 0, 15, self.height())
        painter.drawRect(rect)

    def hide(self):
        self.visible_flag = False
        self.update()

    def show(self):
        self.visible_flag = True
        self.update()

    def resizeEvent(self, event):
        self.update()


class VideoProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Filter Application")
        self.setGeometry(100, 100, 2500, 900)

        # Video capture
        self.video_path = None
        self.cap = None

        # Processed frames
        self.x = None  # Original video frame
        self.filtered_x = None  # Filtered video frame
        self.mask = None  # Difference mask

        # Load default config
        self.config = self.load_config("config.yaml")
        assert self.config, "config.yaml not exists"
        self.config = Option(**self.config)

        # Thread status
        self.is_dividing_scene = False
        self.is_loading_video = []
        self.load_worker_num = self.config.GENERAL_OPTIONS.worker_num
        self.main_ui_lock = threading.Lock()
        self._v_loading_threads = []

        # Filter options
        filter_config = self.config.FILTER_OPTIONS
        self.filter_type = FilterType[filter_config.filter_type]
        self.filter_opts = filter_config.filter_opts

        # UI setup
        self.initUI()

        # Raw Frame list
        self.current_scene = 0
        self.scene_list = [] # [start: end]
        self.scene_divider_dict = {}
        self.thumbnail_dict = {}
        self.score_list = []
        self.batch_size = self.config.GENERAL_OPTIONS.batch_size
        self.current_scroll_start = 0
        self.current_scroll_end = self.batch_size

        # Image list
        self.origin_images = []
        self.filtered_images = []
        self.mask_images = []

        # Timer for video update
        self.current_frame = 0 # frame index of current scene
        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_video)
        self.timer.stop()

        # Key tracker
        self.key_tracker = KeyTracker()
        self.key_tracker.keyPressed.connect(self.handle_key_event)

    def load_config(self, config_file):
        if not osp.exists(config_file):
            return None

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def initUI(self):
        # Main Layout
        layout = QVBoxLayout()

        self.table_splitter = QSplitter(self) # Split video / Scene table
        self.table_splitter.setHandleWidth(1)
        self.table_splitter.setOrientation(Qt.Orientation.Horizontal)

        self.main_splitter = QSplitter(self) # Split video / thumbnail
        self.main_splitter.setHandleWidth(1)
        self.main_splitter.setOrientation(Qt.Orientation.Vertical)

        # Video display
        self.video_widget = QWidget()
        video_layout = QHBoxLayout()

        self.video_sections = []

        self.frame_text = QLabel(f" Scene: None, Frame: None", self)
        self.frame_text.setAlignment(Qt.AlignLeft)
        self.frame_text.setStyleSheet("color: Red; font-size: 18px;")
        self.frame_text.setFixedHeight(20)
        self.frame_text.setMinimumWidth(500)

        for name in ["Original", "Filtered", "Mask"]:
            section = QWidget(self)
            section_layout = QVBoxLayout()

            fold_button = QToolButton(self)
            fold_button.setArrowType(Qt.DownArrow)  # ▼ 기본 아이콘
            fold_button.setCheckable(True)  # 토글 가능
            fold_button.setChecked(False)   # 기본적으로 Label이 보이게 설정

            label = QLabel(self)
            label.setScaledContents(True)  # 크기에 맞춰 조정
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setMinimumSize(50, 50)   # 너무 작아지는 것 방지
            label.setAlignment(Qt.AlignCenter)

            if name == "Original":
                self.label_x = label
            elif name == "Filtered":
                self.label_filtered_x = label
            elif name == "Mask":
                self.label_mask = label

            # 버튼 클릭 시 연결된 QLabel 숨김/표시 & 아이콘 변경
            def toggle_visibility(checked, lbl=label, btn=fold_button):
                lbl.setVisible(not checked)
                btn.setArrowType(Qt.UpArrow if checked else Qt.DownArrow)

            fold_button.toggled.connect(toggle_visibility)

            # 레이아웃 구성
            section_layout.addWidget(fold_button, alignment=Qt.AlignCenter)  # 버튼을 중앙 정렬
            section_layout.addWidget(label)
            section.setLayout(section_layout)

            video_layout.addWidget(section)

            # 저장 (추후 컨트롤을 위해)
            self.video_sections.append((fold_button, label))

        self.video_widget.setLayout(video_layout)
        self.video_widget.setMinimumHeight(400)

        self.main_splitter.addWidget(self.video_widget)

        # Video slider bar (Timeline)
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.sliderMoved.connect(self.on_video_slider_moved)
        self.video_slider.setFixedHeight(30)
        self.main_splitter.addWidget(self.video_slider)

        self.table_splitter.addWidget(self.main_splitter)

        # Scene table
        scene_widget = QWidget(self)
        scene_layout = QVBoxLayout()
        scene_text = QLabel("Scene List")
        scene_layout.addWidget(scene_text)
        self.scene_table = SceneTable(self)
        self.scene_table.setColumnCount(3) # Start, End, Mean_diff
        self.scene_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        scene_layout.addWidget(self.scene_table)

        scene_widget.setLayout(scene_layout)

        self.table_splitter.addWidget(scene_widget)
        self.table_splitter.setSizes([2000, 500])

        layout.addWidget(self.table_splitter)

        # --------- Filter options ---------
        ft_opt_layout = QHBoxLayout()

        # Filter Type
        ft_type_widget = QWidget()
        ft_type_layout = QHBoxLayout()
        ft_combo_text = QLabel("Filter Type: ", self)
        self.ft_type_combo = QComboBox(self)
        self.ft_type_combo.setFixedWidth(150)
        self.ft_type_combo.addItems(["Up-and-Down", "NL-Means", "TComb"])
        self.ft_type_combo.setCurrentIndex(self.filter_type)
        self.ft_type_combo.currentIndexChanged.connect(
            lambda value: self.on_ft_type_changed(value)
        )

        ft_type_layout.addWidget(ft_combo_text)
        ft_type_layout.addWidget(self.ft_type_combo)
        ft_type_widget.setLayout(ft_type_layout)
        ft_type_widget.setMaximumWidth(300)

        ft_opt_layout.addWidget(ft_type_widget)

        self.filter_opt_layouts = {}
        # Down-and-Up
        self.ft_dau_layout = QVBoxLayout()
        self.ft_form_layout = QFormLayout()

        self.ft_dau_mode_combo = QComboBox(self)
        self.ft_dau_mode_combo.setFixedWidth(150)
        self.ft_dau_mode_combo.addItems(["bicubic", "bilinear", "lanczos", "area"])
        self.ft_dau_mode_combo.setCurrentText(self.filter_opts.down_and_up.mode)
        self.ft_dau_mode_combo.currentTextChanged.connect(
            lambda value: self.on_ft_option_changed("down_and_up", "mode", value))

        self.ft_dau_scale_combo = QComboBox(self)
        self.ft_dau_scale_combo.setFixedWidth(150)
        self.ft_dau_scale_combo.addItems(['2', '3', '4', '5', '6'])
        self.ft_dau_scale_combo.setCurrentText(str(self.filter_opts.down_and_up.scale))
        self.ft_dau_scale_combo.currentTextChanged.connect(
            lambda value: self.on_ft_option_changed("down_and_up", "scale", int(value)))

        self.ft_dau_opt_text = QLabel("Down-and-Up filter options")
        self.ft_form_layout.addRow(self.ft_dau_opt_text)
        self.ft_form_layout.addRow("Resize mode", self.ft_dau_mode_combo)
        self.ft_form_layout.addRow("Resize scale", self.ft_dau_scale_combo)
        self.ft_form_layout.setLabelAlignment(Qt.AlignLeft)

        self.ft_dau_layout.addLayout(self.ft_form_layout)
        ft_opt_layout.addLayout(self.ft_dau_layout)

        self.filter_opt_layouts[FilterType.DOWN_AND_UP] = self.ft_dau_layout
        if self.filter_type != FilterType.DOWN_AND_UP:
            self.ft_dau_layout.setEnabled(False)

        # NL denoise
        self.ft_nl_layout = QVBoxLayout()

        ft_opt_layout.addLayout(self.ft_nl_layout)

        self.filter_opt_layouts[FilterType.NL_DENOISE] = self.ft_nl_layout
        if self.filter_type != FilterType.NL_DENOISE:
            self.ft_nl_layout.setEnabled(False)

        # TComb Filter
        self.ft_tcomb_layout = QVBoxLayout()
        self.ft_tcomb_form_layout = QFormLayout()

        # TComb 옵션 추가
        self.ft_tcomb_opt_text = QLabel("TComb filter options")
        self.ft_tcomb_form_layout.addRow(self.ft_tcomb_opt_text)

        self.ft_tcomb_mode_combo = QComboBox(self)
        self.ft_tcomb_mode_combo.setFixedWidth(150)
        self.ft_tcomb_mode_combo.addItems(["Luma Only", "Chroma Only", "Luma+Chroma"])
        self.ft_tcomb_mode_combo.setCurrentIndex(self.filter_opts.tcomb.mode)
        self.ft_tcomb_mode_combo.currentIndexChanged.connect(
            lambda value: self.on_ft_option_changed("tcomb", "mode", value))

        self.ft_tcomb_fthreshl_combo = QComboBox(self)
        self.ft_tcomb_fthreshl_combo.setFixedWidth(150)
        self.ft_tcomb_fthreshl_combo.addItems([str(i) for i in range(1, 17)])
        self.ft_tcomb_fthreshl_combo.setCurrentText(str(self.filter_opts.tcomb.fthreshl))
        self.ft_tcomb_fthreshl_combo.currentTextChanged.connect(
            lambda value: self.on_ft_option_changed("tcomb", "fthreshl", int(value)))

        self.ft_tcomb_fthreshc_combo = QComboBox(self)
        self.ft_tcomb_fthreshc_combo.setFixedWidth(150)
        self.ft_tcomb_fthreshc_combo.addItems([str(i) for i in range(1, 17)])
        self.ft_tcomb_fthreshc_combo.setCurrentText(str(self.filter_opts.tcomb.fthreshc))
        self.ft_tcomb_fthreshc_combo.currentTextChanged.connect(
            lambda value: self.on_ft_option_changed("tcomb", "fthreshc", int(value)))

        self.ft_tcomb_othreshl_combo = QComboBox(self)
        self.ft_tcomb_othreshl_combo.setFixedWidth(150)
        self.ft_tcomb_othreshl_combo.addItems([str(i) for i in range(1, 17)])
        self.ft_tcomb_othreshl_combo.setCurrentText(str(self.filter_opts.tcomb.othreshl))
        self.ft_tcomb_othreshl_combo.currentTextChanged.connect(
            lambda value: self.on_ft_option_changed("tcomb", "othreshl", int(value)))

        self.ft_tcomb_othreshc_combo = QComboBox(self)
        self.ft_tcomb_othreshc_combo.setFixedWidth(150)
        self.ft_tcomb_othreshc_combo.addItems([str(i) for i in range(1, 17)])
        self.ft_tcomb_othreshc_combo.setCurrentText(str(self.filter_opts.tcomb.othreshc))
        self.ft_tcomb_othreshc_combo.currentTextChanged.connect(
            lambda value: self.on_ft_option_changed("tcomb", "othreshc", int(value)))

        self.ft_tcomb_scthresh_combo = QComboBox(self)
        self.ft_tcomb_scthresh_combo.setFixedWidth(150)
        self.ft_tcomb_scthresh_combo.addItems([str(i) for i in range(5, 31, 1)])
        self.ft_tcomb_scthresh_combo.setCurrentText(str(int(self.filter_opts.tcomb.scthresh)))
        self.ft_tcomb_scthresh_combo.currentTextChanged.connect(
            lambda value: self.on_ft_option_changed("tcomb", "scthresh", float(value)))

        self.ft_tcomb_form_layout.addRow("Mode", self.ft_tcomb_mode_combo)
        self.ft_tcomb_form_layout.addRow("Frame Threshold Luma", self.ft_tcomb_fthreshl_combo)
        self.ft_tcomb_form_layout.addRow("Frame Threshold Chroma", self.ft_tcomb_fthreshc_combo)
        self.ft_tcomb_form_layout.addRow("Offset Threshold Luma", self.ft_tcomb_othreshl_combo)
        self.ft_tcomb_form_layout.addRow("Offset Threshold Chroma", self.ft_tcomb_othreshc_combo)
        self.ft_tcomb_form_layout.addRow("Scene Threshold", self.ft_tcomb_scthresh_combo)
        self.ft_tcomb_form_layout.setLabelAlignment(Qt.AlignLeft)

        self.ft_tcomb_layout.addLayout(self.ft_tcomb_form_layout)
        ft_opt_layout.addLayout(self.ft_tcomb_layout)

        self.filter_opt_layouts[FilterType.TCOMB] = self.ft_tcomb_layout
        if self.filter_type != FilterType.TCOMB:
            self.ft_tcomb_layout.setEnabled(False)

        # Bilateral Filter


        layout.addLayout(ft_opt_layout)
        # --------- -------------- ---------


        # Control buttons
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_play)
        button_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)

        # Update filter button
        self.update_filter_button = QPushButton("Update filter", self)
        self.update_filter_button.clicked.connect(self.update_filter_on_video)
        button_layout.addWidget(self.update_filter_button)
        
        # Open video button
        self.open_button = QPushButton("Open Video", self)
        self.open_button.clicked.connect(self.open_video_dialog)
        button_layout.addWidget(self.open_button)

        # Save mask button
        self.save_button = QPushButton("Save Mask", self)
        self.save_button.clicked.connect(self.save_video)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Status bar
        self.status_bar = self.statusBar()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_scene_table(self):
        self.scene_table.blockSignals(True)
        self.scene_table.setRowCount(len(self.scene_list))

        for row, (start, end) in enumerate(self.scene_list):
            self.scene_table.setItem(row, 0, QTableWidgetItem(str(start)))
            self.scene_table.setItem(row, 1, QTableWidgetItem(str(end)))

            mean_score = np.mean(self.score_list[start:end+1])
            score_item = QTableWidgetItem(f"{mean_score:.3f}")
            self.scene_table.setItem(row, 2, score_item)

        self.scene_table.blockSignals(False)  # 이벤트 다시 활성화

        # Display의 Scene text 업데이트
        # TODO: Binary search로 더 빠르게 가능
        for s_i, (start, end) in enumerate(self.scene_list):
            if start <= self.current_frame <= end:
                self.current_scene = s_i
                break

        self.frame_text.setText(
            f"Scene: {self.current_scene+1}/{len(self.scene_list)}, "
            f"Frame: {self.current_frame}")

    def on_video_slider_moved(self, index):
        self.move_to_frame(index)

    def on_video_slider_released(self):
        if self.main_ui_lock.acquire(blocking=False):
            try:
                slider_index = self.video_slider.value()
                if slider_index >= self.frame_len:
                    slider_index = self.frame_len - 1
                self.move_thumbnail_to_index(slider_index)
            finally:
                self.main_ui_lock.release()

    def move_thumbnail_to_index(self, index):
        # 주어진 index가 center가 되도록 thumbnail 업데이트
        new_scroll_end = min(index + self.batch_size, self.frame_len)
        new_scroll_start = max(0, index - self.batch_size)

        # 우측 프레임 추가
        if new_scroll_end > self.current_scroll_end:
            self.load_thumbnail(new_scroll_start, new_scroll_end - new_scroll_start)
            self.remove_old_thumbnails(new_scroll_start, new_scroll_end)

        # 좌측 프레임 추가
        elif new_scroll_start < self.current_scroll_start:
            self.load_thumbnail(new_scroll_start, new_scroll_end - new_scroll_start, reverse=True)
            self.remove_old_thumbnails(new_scroll_start, new_scroll_end)

        self.current_scroll_start = new_scroll_start
        self.current_scroll_end = new_scroll_end

        self.locate_thumbnail_center(index)

    def locate_thumbnail_center(self, index):
        # UI 업데이트 전에 호출하면 썸네일 좌표 계산이 제대로 동작하지 않음
        self.thumb_layout.activate() # 썸네일 영역 강제 업데이트 요청
        QApplication.processEvents() # 이벤트 루프 실행하여 UI 업데이트 강제 적용
        """index 위치의 썸네일을 중앙에 오도록 스크롤 이동"""
        scroll_bar = self.thumbnail_area.horizontalScrollBar()
        thumb_widget = self.thumbnail_dict[index]
        widget_left = thumb_widget.pos().x()
        widget_width = thumb_widget.width()
        scroll_width = self.thumbnail_area.width()
        target_scroll_pos = widget_left + (widget_width // 2) - (scroll_width // 2)
        scroll_bar.setValue(target_scroll_pos)

    def on_thumb_slider_released(self):
        """썸네일 스크롤을 release 했을 때, 인접 프레임을 더 가져오는 함수"""
        scrollbar = self.thumbnail_area.horizontalScrollBar()
        min_scroll = scrollbar.minimum()
        max_scroll = scrollbar.maximum()
        current_scroll = scrollbar.value()

        # 우측: 새 데이터 로딩
        if current_scroll == max_scroll and self.current_scroll_end < self.frame_len:
            prev_scroll_end = self.current_scroll_end
            self.current_scroll_end = min(self.current_scroll_end + self.batch_size, self.frame_len)
            self.current_scroll_start = max(self.current_scroll_end - (self.batch_size * 2), 0)
            load_len = self.current_scroll_end - self.current_scroll_start
            self.load_thumbnail(self.current_scroll_start, load_len)
            self.remove_old_thumbnails(self.current_scroll_start, self.current_scroll_end)

            self.locate_thumbnail_center(prev_scroll_end)

        # 좌측: 이전 데이터 로딩
        elif current_scroll == min_scroll and self.current_scroll_start > 0:
            prev_scroll_start = self.current_scroll_start
            self.current_scroll_start = max(self.current_scroll_start - self.batch_size, 0)
            self.current_scroll_end = min(self.current_scroll_start + (self.batch_size * 2), self.frame_len)
            load_len = self.current_scroll_end - self.current_scroll_start
            self.load_thumbnail(self.current_scroll_start, load_len, reverse=True)
            self.remove_old_thumbnails(self.current_scroll_start, self.current_scroll_end)

            self.locate_thumbnail_center(prev_scroll_start)

    def add_scene(self):
        if self.video_path is None:
            return

        new_start = self.scene_list[-1][1]
        new_end = new_start
        self.scene_list.append([new_start, new_end])
        self.load_scene_table()

    def remove_scene(self):
        selected_row = self.scene_table.currentRow()
        if selected_row >= 0:
            del self.scene_list[selected_row]
        self.load_scene_table()


    def load_frame_data(self, index):
        if index >= self.batch_size and not self.is_first_batch_loaded:
            self.load_thumbnail(self.current_scroll_start, self.batch_size)
            self.is_first_batch_loaded = True

    def load_thumbnail(self, start_index, count, reverse=False):
        """Load frames to thumbnail area.
        If reverse is True, it appends frames to the left side
        """
        end_index = min(start_index + count, self.frame_len)

        _range = reversed(range(start_index, end_index)) if reverse else range(start_index, end_index)

        for index in _range:
            if index in self.thumbnail_dict:
                continue
            ori_image = self.origin_images[index]
            fil_image = self.filtered_images[index]
            mask = self.mask_images[index]
            score = self.score_list[index]

            thumbnail = QWidget()
            frame_group_layout = QVBoxLayout()
            original_qimage = QImage(ori_image.data, ori_image.shape[1], ori_image.shape[0], 3 * ori_image.shape[1], QImage.Format_RGB888)
            original_pixmap = QPixmap(original_qimage).scaled(100, 100, Qt.KeepAspectRatio)
            filtered_qimage = QImage(fil_image.data, fil_image.shape[1], fil_image.shape[0], 3 * fil_image.shape[1], QImage.Format_RGB888)
            filtered_pixmap = QPixmap(filtered_qimage).scaled(100, 100, Qt.KeepAspectRatio)
            mask_qimage = QImage(mask.data, mask.shape[1], mask.shape[0], mask.shape[1], QImage.Format_Grayscale8)
            mask_pixmap = QPixmap(mask_qimage).scaled(100, 100, Qt.KeepAspectRatio)

            # Original image thumbnail
            original_thumb_label = QLabel(self)
            original_thumb_label.setPixmap(original_pixmap)
            original_thumb_label.setAlignment(Qt.AlignCenter)
            original_thumb_label.mousePressEvent = lambda event, i=index: self.handle_thumb_mouse_event(event, i)

            original_text = QLabel(f"{index}: {score:.4f}", self)
            original_text.setAlignment(Qt.AlignCenter)
            original_text.setStyleSheet("color: white; background: rgba(0, 0, 0, 128); font-size: 10px;")
            original_text.setFixedHeight(10)

            frame_group_layout.addWidget(original_text)
            frame_group_layout.addWidget(original_thumb_label)

            # Filtered image thumbnail
            filtered_thumb_label = QLabel(self)
            filtered_thumb_label.setPixmap(filtered_pixmap)
            filtered_thumb_label.setAlignment(Qt.AlignCenter)
            filtered_thumb_label.mousePressEvent = lambda event, i=index: self.handle_thumb_mouse_event(event, i)

            frame_group_layout.addWidget(filtered_thumb_label)

            # Mask image thumbnail
            mask_thumb_label = QLabel(self)
            mask_thumb_label.setPixmap(mask_pixmap)
            mask_thumb_label.setAlignment(Qt.AlignCenter)
            mask_thumb_label.mousePressEvent = lambda event, i=index: self.handle_thumb_mouse_event(event, i)

            frame_group_layout.addWidget(mask_thumb_label)
            thumbnail.setLayout(frame_group_layout)
            self.thumbnail_dict[index] = thumbnail

            divider = DividerWidget()
            divider.mousePressEvent = lambda event, i=index: self.handle_divider_mouse_event(event, i)
            self.scene_divider_dict[index] = divider
            if reverse:
                self.thumb_layout.insertWidget(0, divider)
                self.thumb_layout.insertWidget(0, thumbnail)
            else:
                self.thumb_layout.addWidget(thumbnail)
                self.thumb_layout.addWidget(divider)

            divider.hide()
            for _, end in self.scene_list:
                if index == end:
                    divider.show()

    def remove_old_thumbnails(self, keep_start, keep_end):
        keys_to_remove = [i for i in list(self.thumbnail_dict.keys()) if i < keep_start or i >= keep_end]
        for i in keys_to_remove:
            thumbnail = self.thumbnail_dict.pop(i)
            self.thumb_layout.removeWidget(thumbnail)
            thumbnail.deleteLater()

            divider = self.scene_divider_dict.pop(i)
            self.thumb_layout.removeWidget(divider)
            divider.deleteLater()


    def finish_load_frame_data(self, i):
        self.is_loading_video[i] = False

        if not self.is_dividing_scene and not any(self.is_loading_video):
            # Update divider
            for _, end in self.scene_list:
                if end in self.scene_divider_dict:
                    self.scene_divider_dict[end].show()

    def load_video(self):
        # Remove previous thumbnail
        if hasattr(self, 'thumbnail_area'):
            self.thumbnail_area.deleteLater()

        self.thumb_layout = QHBoxLayout()
        self.thumbnail_area = QScrollArea(self)
        self.thumbnail_area.setWidgetResizable(True)
        scroll_area_widget = QWidget(self)
        scroll_area_widget.setLayout(self.thumb_layout)
        self.thumbnail_area.setWidget(scroll_area_widget)
        scroll_area_widget.adjustSize()
        self.main_splitter.addWidget(self.thumbnail_area)

        # Virtual scroll
        self.thumbnail_area.horizontalScrollBar().sliderReleased.connect(self.on_thumb_slider_released)
        self.current_scroll_start = 0
        self.current_scroll_end = min(self.batch_size, self.frame_len)

        self.update_filter_on_video()

    def move_to_next_scene(self):
        if self.main_ui_lock.acquire(blocking=False):
            try:
                if len(self.scene_list) == 0:
                    return
                next_scene = (self.current_scene + 1) % len(self.scene_list)
                next_frame = self.scene_list[next_scene][0]
                self.move_to_frame(next_frame)
                self.move_thumbnail_to_index(next_frame)
            finally:
                self.main_ui_lock.release()

    def move_to_prev_scene(self):
        if self.main_ui_lock.acquire(blocking=False):
            try:
                if len(self.scene_list) == 0:
                    return
                prev_scene = (self.current_scene - 1) % len(self.scene_list)
                prev_frame = self.scene_list[prev_scene][0]
                self.move_to_frame(prev_frame)
                self.move_thumbnail_to_index(prev_frame)
            finally:
                self.main_ui_lock.release()

    def move_to_next_frame(self):
        next_frame = (self.current_frame + 1) % self.frame_len
        self.move_to_frame(next_frame)

    def move_to_prev_frame(self):
        prev_frame = (self.current_frame - 1) % self.frame_len
        self.move_to_frame(prev_frame)

    def move_to_frame(self, index):
        if len(self.origin_images) <= index:
            return

        if not self.is_loaded[index]:
            # 인접 프레임 처리하도록 요청
            # if self._v_loading_threads:
            #     self._v_loading_threads[0].request_frame(index)
            #     if not self._v_loading_threads[0].running:
            #         self._v_loading_threads[0].start()

            # 현재 프레임은 직접 로드
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, frame = self.cap.read()
            ori_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fil_image = apply_filter(ori_image, self.filter_type, self.filter_opts)
            mask = get_mask(ori_image, fil_image)
        else:
            ori_image, fil_image, mask = self.origin_images[index], self.filtered_images[index], self.mask_images[index]

        self.current_frame = index
        # TODO: Binary search로 더 빠르게 가능
        for s_i, (start, end) in enumerate(self.scene_list):
            if start <= index <= end:
                self.current_scene = s_i

        self.frame_text.setText(
            f"Scene: {self.current_scene+1}/{len(self.scene_list)}, "
            f"Frame: {self.current_frame}")

        ori_qimage = QImage(ori_image.data, ori_image.shape[1], ori_image.shape[0], 3 * ori_image.shape[1], QImage.Format_RGB888)
        fil_qimage = QImage(fil_image.data, fil_image.shape[1], fil_image.shape[0], 3 * fil_image.shape[1], QImage.Format_RGB888)
        mask_qimage = QImage(mask.data, mask.shape[1], mask.shape[0], mask.shape[1], QImage.Format_Grayscale8)

        self.label_x.setPixmap(QPixmap.fromImage(ori_qimage))
        self.label_filtered_x.setPixmap(QPixmap.fromImage(fil_qimage))
        self.label_mask.setPixmap(QPixmap.fromImage(mask_qimage))
        self.label_x.mousePressEvent = lambda event, i=index: self.move_thumbnail_to_index(i)
        self.label_filtered_x.mousePressEvent = lambda event, i=index: self.move_thumbnail_to_index(i)
        self.label_mask.mousePressEvent = lambda event, i=index: self.move_thumbnail_to_index(i)
        self.video_slider.setValue(index)

    def toggle_play(self):
        if self.video_path is not None:
            self.timer.start(30)
            self.is_playing = True

    def toggle_pause(self):
        self.timer.stop()
        self.is_playing = False

    def play_video(self):
        if len(self.origin_images) == 0:
            return

        self.current_frame = (self.current_frame + 1) % self.frame_len
        self.move_to_frame(self.current_frame)

    def start_divide_scene(self, video_path):
        self.is_dividing_scene = True
        self._s_divide_thread = SceneDivideThread(video_path,
                                                  start_frame=0,
                                                  detect_model=self.config.SCENE_DIVIDE_OPTIONS.model_name)
        self._s_divide_thread.stdout_signal.connect(self.update_statusbar)
        self._s_divide_thread.result_signal.connect(self.finish_divide_scene)
        self._s_divide_thread.start()

    def finish_divide_scene(self, scene_list):
        self.scene_list = scene_list
        self.is_dividing_scene = False
        self.load_scene_table()

        # 씬 탐지가 완료되면 필터링 자동 시작
        self.update_statusbar("씬 탐지 완료, 필터링을 시작합니다...")
        self.update_filter_on_video()

        if not any(self.is_loading_video):
            # Update divider
            for _, end in self.scene_list:
                if end in self.scene_divider_dict:
                    self.scene_divider_dict[end].show()

    def init_video_data(self, video_path):
        if self.cap is not None:
            self.cap.release()

        # Check video file can be opened
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")

        self.video_path = video_path
        self.frame_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Timeline slider setting
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(self.frame_len)
        self.video_slider.setValue(0)
        self.video_slider.setSingleStep(1)
        self.video_slider.sliderReleased.connect(self.on_video_slider_released)

        try:
            self.origin_images = np.memmap(osp.join(TEMP_DIR, 'ori_images.dat'), dtype=np.uint8, mode='w+',
                                           shape=(self.frame_len, self.frame_height, self.frame_width, 3))
            self.filtered_images = np.memmap(osp.join(TEMP_DIR, 'fil_images.dat'), dtype=np.uint8, mode='w+',
                                             shape=(self.frame_len, self.frame_height, self.frame_width, 3))
            self.mask_images = np.memmap(osp.join(TEMP_DIR, 'mask_images.dat'), dtype=np.uint8, mode='w+',
                                         shape=(self.frame_len, self.frame_height, self.frame_width))
        except OSError as e:
            self.update_statusbar(f"Error: Unable to allocate memory on disk. {e}")
        self.score_list = np.zeros(self.frame_len, dtype=np.float64)
        self.is_loaded = np.zeros(self.frame_len, dtype=np.bool) # Whether the frame data is loaded

        self.current_scene = 0
        self.scene_list = [] # [[start: end], ...]
        self.scene_divider_dict = {}
        self.thumbnail_dict = {}
        self.current_frame = 0
        self.is_first_batch_loaded = False

    def open_video_dialog(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if video_path:
            self.open_video(video_path)

    def open_video(self, video_path):
        try:
            self.init_video_data(video_path)
        except ValueError as e:
            self.update_statusbar(e)
            return

        self.start_divide_scene(video_path)
        self.load_video()
        self.move_to_frame(0)

    def get_video_codec_and_format(self):
        if not self.video_path:
            return

        ffprobe_command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "default=noprint_wrappers=1"
        ]

        ffprobe_command.append(self.video_path)
        result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        codec_info = result.stdout.split('\n')
        codec_name = codec_info[0].split('=')[1] if len(codec_info) > 0 else "unknown"
        pix_fmt = codec_info[1].split('=')[1] if len(codec_info) > 1 else "unknown"

        return codec_name, pix_fmt

    def save_video(self):
        if not self.video_path:
            self.status_bar.showMessage("Video is not loaded.")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Select Directory to save Video")

        if save_path:
            codec_name, pix_fmt = self.get_video_codec_and_format()
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            ffmpeg_command = [
                "ffmpeg",
                "-f", "rawvideo",            # rawvideo 형식으로 입력
                "-pix_fmt", "gray",          # 그레이스케일 영상이므로 'gray'로 설정
                "-s", f"{self.frame_width}x{self.frame_height}",            # 해상도 (W x H)
                "-r", str(fps),              # FPS
                "-i", "-",                   # 입력은 stdin
                "-c:v", codec_name,
                "-pix_fmt", pix_fmt,
                save_path
            ]
            process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
            for i in range(self.frame_len):
                frame = self.mask_images[i]
                # 그레이스케일 프레임을 raw 데이터로 전달
                process.stdin.write(frame.tobytes())

            process.stdin.close()
            process.wait()
            self.update_statusbar(f"Mask saved in {save_path}")

    def on_ft_type_changed(self, value):
        self.filter_type = value
        for type, layout in self.filter_opt_layouts.items():
            layout.setEnabled(type == value)

    def on_ft_option_changed(self, ft_type, option_name, value):
        self.filter_opts[ft_type][option_name] = value

    def update_filter_on_video(self):
        """필터를 적용"""
        if not self.video_path:
            return

        # 필터가 TComb인 경우 처리 방식
        if self.filter_type == FilterType.TCOMB and len(self.scene_list) > 0:
            # 기존 스레드 정리
            if hasattr(self, '_tcomb_thread') and self._tcomb_thread is not None:
                if self._tcomb_thread.isRunning():
                    self._tcomb_thread.stop()
                    self._tcomb_thread.wait()
                    self._tcomb_thread.deleteLater()
                    
            # 전체 씬에 대해 순차적으로 처리하기 위한 변수들
            if not hasattr(self, 'current_processing_scene') or self.current_processing_scene is None:
                self.current_processing_scene = 0
                self.process_all_scenes = True
            
            # 현재 처리할 씬 정보 가져오기
            if self.current_processing_scene < len(self.scene_list):
                start_frame, end_frame = self.scene_list[self.current_processing_scene]
                scene_name = f"씬 {self.current_processing_scene+1}/{len(self.scene_list)}"
                
                self.update_statusbar(f"{scene_name} (프레임 {start_frame}~{end_frame})에 TComb 필터를 적용합니다...")
                
                # TComb 스레드 생성 및 실행
                self._tcomb_thread = TCombFilterThread(
                    self.video_path,
                    self.filter_opts,
                    self.origin_images,
                    self.filtered_images,
                    self.mask_images,
                    self.score_list,
                    self.is_loaded,
                    start_frame=start_frame,
                    end_frame=end_frame
                )
                
                # 진행 상황 연결
                self._tcomb_thread.progress_signal.connect(self.update_tcomb_progress)
                
                # 완료 시그널 연결 - 전체 처리 모드일 경우 다음 씬으로 넘어가는 로직 추가
                self._tcomb_thread.result_signal.connect(
                    lambda: self.finish_tcomb_processing(scene_name, start_frame, process_next=self.process_all_scenes)
                )
                
                # TComb 처리 시작
                self._tcomb_thread.start()
                return
            else:
                # 모든 씬 처리 완료
                self.current_processing_scene = None
                self.process_all_scenes = False
                self.update_statusbar("모든 씬에 대한 TComb 처리가 완료되었습니다.")
                return
        
        # 다른 필터 타입이거나 씬이 없는 경우 기존 방식대로 처리
        # 이전에 실행중이던 스레드가 있으면 정지 및 리소스 정리
        if self._v_loading_threads:
            for t in self._v_loading_threads:
                if t.isRunning():
                    t.stop()
                    t.wait()
                    t.deleteLater()
            self._v_loading_threads = []

        # score 및 load 상황 초기화
        self.score_list = np.zeros(self.frame_len, dtype=np.float64)
        self.is_loaded = np.zeros(self.frame_len, dtype=np.bool)

        worker_num = self.load_worker_num if self.frame_len > self.load_worker_num else 1
        self.is_loading_video = [True for _ in range(worker_num)]

        def split_indices(N, k):
            sizes = [N // k + (i < N % k) for i in range(k)]
            return [[sum(sizes[:i]), sum(sizes[:i+1])] for i in range(k)]

        splited_indices = split_indices(self.frame_len, worker_num)
        self._v_loading_threads = []
        for i, (s, e) in enumerate(splited_indices):
            v_loading_thread = VideoLoadAndFilterThread(self.video_path,
                                                       self.filter_type,
                                                       self.filter_opts,
                                                       self.origin_images,
                                                       self.filtered_images,
                                                       self.mask_images,
                                                       self.score_list,
                                                       self.is_loaded,
                                                       start_index=s,
                                                       end_index=e)
            v_loading_thread.frame_signal.connect(self.load_frame_data)
            v_loading_thread.result_signal.connect(lambda: self.finish_load_frame_data(i))
            self._v_loading_threads.append(v_loading_thread)
            v_loading_thread.start()

        # 현재 로딩 진행 상황 Timer
        self.load_start_time = time.perf_counter()
        if hasattr(self, 'load_timer') and self.load_timer.isActive():
            self.load_timer.stop()
        self.load_timer = QTimer(self)
        self.load_timer.timeout.connect(self.update_load_status)
        self.load_timer.start(1000) # 1000ms = 1초

        # 현재 프레임 업데이트
        cf = self.current_frame
        self.move_to_frame(cf)

    def update_load_status(self):
        load_count = self.is_loaded.sum()
        if load_count == self.frame_len:
            self.load_timer.stop()
            elapsed_time = round(time.perf_counter() - self.load_start_time, 3)
            self.update_statusbar(f"All frames are processed. elapsed time: {elapsed_time}")

            # Scene table도 다시 업데이트
            self.load_scene_table()
        else:
            self.update_statusbar(f"Processed frames: {load_count} / {self.frame_len}")

    def update_statusbar(self, message):
        self.status_bar.showMessage(message)

    @pyqtSlot(int)
    def handle_key_event(self, key):
        if key == Qt.Key_Space:
            if self.is_playing:
                self.toggle_pause()
            else:
                self.toggle_play()
        elif key == Qt.Key_Up:
            if self.video_path is not None:
                self.move_to_next_scene()
        elif key == Qt.Key_Down:
            if self.video_path is not None:
                self.move_to_prev_scene()
        elif key == Qt.Key_Left:
            if self.video_path is not None:
                self.move_to_prev_frame()
        elif key == Qt.Key_Right:
            if self.video_path is not None:
                self.move_to_next_frame()

    def handle_thumb_mouse_event(self, event, index):
        if event.buttons() & Qt.LeftButton:
            self.move_to_frame(index)


    def handle_divider_mouse_event(self, event, index):
        if event.buttons() & Qt.LeftButton:
            divider = self.scene_divider_dict[index]
            # Merge scene list
            if divider.visible_flag:
                for i, (start, end) in enumerate(self.scene_list):
                    if end == index:
                        if i + 1 < len(self.scene_list):
                            next_start, next_end = self.scene_list[i + 1]
                            merged = [start, next_end]
                            self.scene_list[i] = merged
                            del self.scene_list[i + 1]
                divider.hide()
            else:
                new_scene_list = []
                for start, end in self.scene_list:
                    if start <= index <= end:
                        new_scene_list.append([start, index])
                        new_scene_list.append([index + 1, end])
                    else:
                        new_scene_list.append([start, end])
                    self.scene_list = new_scene_list
                divider.show()
            self.load_scene_table()

    def update_tcomb_progress(self, progress):
        """TComb 처리 진행 상황 업데이트"""
        self.update_statusbar(f"TComb 처리 중: {progress}%")
        
    def finish_tcomb_processing(self, scene_name, start_frame, process_next=False):
        """TComb 처리 완료 후 처리"""
        self.update_statusbar(f"{scene_name}의 TComb 처리가 완료되었습니다.")
        
        # 현재 프레임 위치 유지 (자동으로 이동하지 않음)
        # 현재 프레임이 처리된 씬 내에 있을 경우만 해당 프레임을 다시 로드하여 결과 보여주기
        if hasattr(self, 'current_processing_scene'):
            current_scene_start, current_scene_end = self.scene_list[self.current_processing_scene]
            if current_scene_start <= self.current_frame <= current_scene_end:
                self.move_to_frame(self.current_frame)
        
        # 메모리 정리
        if hasattr(self, '_tcomb_thread') and self._tcomb_thread is not None:
            self._tcomb_thread.wait(1000)  # 스레드가 완전히 종료될 때까지 최대 1초 대기
            self._tcomb_thread.deleteLater()
            self._tcomb_thread = None
            
        # 전체 처리 모드에서 다음 씬 처리
        if process_next and self.process_all_scenes:
            self.current_processing_scene += 1
            # 일정 시간 후 다음 씬 처리 시작
            QTimer.singleShot(1500, self.update_filter_on_video)

    def __del__(self):
        """객체 소멸 시 리소스 정리"""
        self.cleanup_resources()
        
    def cleanup_resources(self):
        """모든 리소스 정리"""
        # 타이머 정지
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
            
        if hasattr(self, 'load_timer') and self.load_timer.isActive():
            self.load_timer.stop()
        
        # TComb 스레드 정리
        if hasattr(self, '_tcomb_thread') and self._tcomb_thread is not None:
            if self._tcomb_thread.isRunning():
                self._tcomb_thread.stop()
                self._tcomb_thread.wait(2000)  # 최대 2초 대기
                
        # 비디오 로딩 스레드 정리
        if hasattr(self, '_v_loading_threads') and self._v_loading_threads:
            for t in self._v_loading_threads:
                if t and t.isRunning():
                    t.stop()
                    t.wait(1000)  # 최대 1초 대기
            
        # 씬 디바이드 스레드 정리
        if hasattr(self, '_s_divide_thread') and self._s_divide_thread is not None:
            if self._s_divide_thread.isRunning():
                self._s_divide_thread.terminate()
                self._s_divide_thread.wait(1000)  # 최대 1초 대기
        
        # 메모리매핑 파일 정리
        if hasattr(self, 'origin_images') and isinstance(self.origin_images, np.memmap):
            del self.origin_images
        if hasattr(self, 'filtered_images') and isinstance(self.filtered_images, np.memmap):
            del self.filtered_images
        if hasattr(self, 'mask_images') and isinstance(self.mask_images, np.memmap):
            del self.mask_images
            
        # 비디오 캡처 객체 닫기
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            
        # multiprocessing 큐 정리
        for thread in getattr(self, '_v_loading_threads', []):
            if hasattr(thread, 'request_queue'):
                try:
                    # 큐 비우기
                    while not thread.request_queue.empty():
                        thread.request_queue.get_nowait()
                    thread.request_queue.close()
                    thread.request_queue.join_thread()
                except:
                    pass
        
        # 개방된 파일 디스크립터 정리를 위한 가비지 컬렉션 실행
        import gc
        gc.collect()
        
    def closeEvent(self, event):
        """창을 닫을 때 리소스 정리"""
        self.cleanup_resources()
        
        # 모든 스레드가 종료될 때까지 짧게 대기
        QApplication.processEvents()
        
        super().closeEvent(event)


def delete_memmap_file():
    file_names = ["ori_images.dat", "fil_images.dat", "mask_images.dat"]
    for fn in file_names:
        memmap_file = os.path.join(TEMP_DIR, fn)
        if os.path.exists(memmap_file):
            try:
                os.remove(memmap_file)
            except Exception as e:
                print(f"메모리 매핑 파일 삭제 실패: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = VideoProcessor()
    window.show()

    atexit.register(delete_memmap_file)

    app.exec_()
