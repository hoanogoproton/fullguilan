import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtGui, QtCore
from ultralytics.utils.plotting import Annotator, colors
from PyQt5.QtGui import QPixmap
from guicam import Ui_MainWindow
from ultralytics import YOLO
import cv2
import numpy as np


class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)

        # Camera state
        self.camera_running_1 = False
        self.camera_running_2 = False
        self.cap_1 = None
        self.cap_2 = None

        # Load YOLO models
        self.model_1 = YOLO("thanlanquan.pt")
        self.model_2 = YOLO("mamlan.pt")

        # Connect buttons for Model 1
        self.uic.Browser_button.clicked.connect(self.linkto_1)
        self.uic.start_camera_button.clicked.connect(self.start_camera_1)
        self.uic.stop_camera_button.clicked.connect(self.stop_camera_1)

        # Connect buttons for Model 2
        self.uic.Browser_button_2.clicked.connect(self.linkto_2)
        self.uic.start_camera_button_a.clicked.connect(self.start_camera_2)
        self.uic.stop_camera_button_a.clicked.connect(self.stop_camera_2)

    # ---------------------- Model 1 Functions ----------------------

    def linkto_1(self):
        link1 = QFileDialog.getOpenFileName(filter='*.png *.jpg *.jpeg')
        self.uic.lineEdit.setText(link1[0])
        global image_1
        image_1 = cv2.imread(link1[0])

        if image_1 is None:
            self.uic.lineEdit.setText("Invalid image!")
            return

        self.process_image_1(image_1)

    def start_camera_1(self):
        if self.camera_running_1:
            return

        self.camera_running_1 = True
        self.cap_1 = cv2.VideoCapture(0)

        if not self.cap_1.isOpened():
            self.uic.lineEdit.setText("Cannot open camera!")
            self.camera_running_1 = False
            return

        self.timer_1 = QtCore.QTimer()
        self.timer_1.timeout.connect(self.update_frame_1)
        self.timer_1.start(30)

    def stop_camera_1(self):
        if not self.camera_running_1:
            return

        self.camera_running_1 = False
        self.timer_1.stop()
        self.cap_1.release()
        self.uic.Screen.clear()

    def update_frame_1(self):
        ret, frame = self.cap_1.read()
        if not ret:
            self.uic.lineEdit.setText("Failed to grab frame!")
            return

        self.process_image_1(frame)

    def process_image_1(self, image):
        results = self.model_1(image)
        result = results[0]

        class_count = {}
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = self.model_1.names[class_id]
            class_count[label] = class_count.get(label, 0) + 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label_text = f"{label} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        self.update_class_info(class_count, self.uic.name_1, self.uic.number_1)
        self.display_image(image, self.uic.Screen)

    # ---------------------- Model 2 Functions ----------------------

    def linkto_2(self):
        link2 = QFileDialog.getOpenFileName(filter='*.png *.jpg *.jpeg')
        self.uic.lineEdit_2.setText(link2[0])
        global image_2
        image_2 = cv2.imread(link2[0])

        if image_2 is None:
            self.uic.lineEdit_2.setText("Invalid image!")
            return

        self.process_image_2(image_2)

    def start_camera_2(self):
        if self.camera_running_2:
            return

        self.camera_running_2 = True
        self.cap_2 = cv2.VideoCapture(0)

        if not self.cap_2.isOpened():
            self.uic.lineEdit_2.setText("Cannot open camera!")
            self.camera_running_2 = False
            return

        self.timer_2 = QtCore.QTimer()
        self.timer_2.timeout.connect(self.update_frame_2)
        self.timer_2.start(30)

    def stop_camera_2(self):
        if not self.camera_running_2:
            return

        self.camera_running_2 = False
        self.timer_2.stop()
        self.cap_2.release()
        self.uic.Screen_2.clear()

    def update_frame_2(self):
        ret, frame = self.cap_2.read()
        if not ret:
            self.uic.lineEdit_2.setText("Failed to grab frame!")
            return

        self.process_image_2(frame)

    def process_image_2(self, image):
        # Dự đoán
        results2 = self.model_2(image)
        result2 = results2[0]

        # Lấy tên lớp từ mô hình
        class_names2 = self.model_2.names if hasattr(self.model_2, 'names') else []
        class_count = {}

        # Khởi tạo Annotator để xử lý vẽ elip
        annotator = Annotator(image, line_width=2)

        if result2.masks is not None:
            # Duyệt qua từng mask
            clss = result2.boxes.cls.cpu().tolist()
            confs = result2.boxes.conf.cpu().tolist()
            masks = result2.masks.xy

            for mask, cls, conf in zip(masks, clss, confs):
                if conf < 0.5:  # Bỏ qua các đối tượng có độ chính xác nhỏ hơn 0.5
                    continue

                # Tăng số lượng đối tượng của lớp
                cls_name = class_names2[int(cls)]
                class_count[cls_name] = class_count.get(cls_name, 0) + 1

                # Vẽ kết quả lên ảnh
                points = np.array(mask, dtype=np.int32)
                color = colors(int(cls), True)
                txt_color = annotator.get_txt_color(color)

                if points.shape[0] >= 5:  # fitEllipse yêu cầu >= 5 điểm
                    ellipse = cv2.fitEllipse(points)

                    # Vẽ elip
                    cv2.ellipse(image, ellipse, color, 2)

                    # Vẽ trục đối xứng (trục nhỏ của elip)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    axis_minor = min(ellipse[1]) / 2  # Trục nhỏ / 2
                    angle = ellipse[2]

                    # Tính hai đầu mút của trục nhỏ
                    x1 = int(center[0] + axis_minor * np.cos(np.radians(angle + 90)))
                    y1 = int(center[1] + axis_minor * np.sin(np.radians(angle + 90)))
                    x2 = int(center[0] - axis_minor * np.cos(np.radians(angle + 90)))
                    y2 = int(center[1] - axis_minor * np.sin(np.radians(angle + 90)))

                    # Vẽ trục nhỏ
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Màu xanh dương cho trục nhỏ

                    # Vẽ nhãn với độ chính xác
                    label = f"{cls_name}: {conf:.2f}"
                    annotator.text((center[0], center[1]), label, txt_color=(255, 255, 255))

        # Cập nhật thông tin lớp và hiển thị ảnh
        self.update_class_info(class_count, self.uic.name_2, self.uic.number_2)
        self.display_image(image, self.uic.Screen_2)

    # ---------------------- Shared Functions ----------------------

    def display_image(self, image, screen):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        screen.setPixmap(QPixmap.fromImage(q_image))

    def update_class_info(self, class_count, name_label, number_label):
        if class_count:
            max_class = max(class_count, key=class_count.get)
            max_count = class_count[max_class]
        else:
            max_class = "No objects detected"
            max_count = 0

        name_label.setText(max_class)
        number_label.setText(str(max_count))

    def show(self):
        self.main_win.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
