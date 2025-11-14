import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QHBoxLayout,
    QMessageBox,
    QProgressBar
)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from aa import load_image, save_image, style_transfer  # 导入现有代码中的函数
import os
import shutil

# 添加资源路径处理函数
def resource_path(relative_path):
    """ 获取资源文件的绝对路径，适配开发和打包环境 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 会将资源文件解压到 _MEIPASS 目录
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class StyleTransferWorker(QThread):
    """后台线程用于执行风格迁移"""
    finished = pyqtSignal(object)  # 定义信号，用于通知主线程任务完成
    progress = pyqtSignal(int)  # 添加进度信号
    
    def __init__(self, content_path, style_path):
        super().__init__()
        self.content_path = content_path
        self.style_path = style_path
        
    def run(self):
        # 加载图片并进行风格迁移
        content_tensor = load_image(self.content_path)
        style_tensor = load_image(self.style_path)
        # 修改style_transfer调用，添加进度回调
        result_tensor = style_transfer(
            content_tensor, 
            style_tensor, 
            progress_callback=self.progress.emit
        )

        # 保存结果图片，确保路径正确
        output_path = resource_path("output.jpg")
        save_image(result_tensor, output_path)
        self.finished.emit(output_path)


class StyleTransferApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("风格迁移应用")
        self.setGeometry(100, 100, 900, 700)

        # 设置全局样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel#result_label {
                background-color: #fff;
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                min-height: 200px;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                width: 20px;
            }
        """)

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        # 主布局：垂直布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setAlignment(Qt.AlignCenter)

        # 创建三列布局
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.setAlignment(Qt.AlignCenter)
        self.bottom_layout.setSpacing(30)

        # 设置三个主要部分
        self.setup_content_section()
        self.setup_style_section()
        self.setup_result_section()

        self.main_layout.addLayout(self.bottom_layout)

        # 初始化图片路径
        self.content_path = None
        self.style_path = None
        self.result_path = None

    def setup_content_section(self):
        content_layout = QVBoxLayout()
        content_layout.setAlignment(Qt.AlignCenter)

        # 内容图片标签和按钮
        self.content_label = QLabel("内容图片：未选择")
        self.content_button = QPushButton("选择内容图片")
        self.content_button.setIcon(QIcon(resource_path("icons/image.png")))
        
        # 内容图片预览区域
        self.content_image_display = QLabel("内容图片预览")
        self.content_image_display.setObjectName("result_label")
        self.content_image_display.setAlignment(Qt.AlignCenter)
        self.content_image_display.setFixedSize(300, 300)

        # 添加到布局
        content_layout.addWidget(self.content_label)
        content_layout.addWidget(self.content_button)
        content_layout.addWidget(self.content_image_display)
        self.bottom_layout.addLayout(content_layout)

    def setup_style_section(self):
        style_layout = QVBoxLayout()
        style_layout.setAlignment(Qt.AlignCenter)

        # 风格图片标签和按钮
        self.style_label = QLabel("风格图片：未选择")
        self.style_button = QPushButton("选择风格图片")
        self.style_button.setIcon(QIcon(resource_path("icons/palette.png")))

        # 风格图片预览区域
        self.style_image_display = QLabel("风格图片预览")
        self.style_image_display.setObjectName("result_label")
        self.style_image_display.setAlignment(Qt.AlignCenter)
        self.style_image_display.setFixedSize(300, 300)

        # 添加到布局
        style_layout.addWidget(self.style_label)
        style_layout.addWidget(self.style_button)
        style_layout.addWidget(self.style_image_display)
        self.bottom_layout.addLayout(style_layout)

    def setup_result_section(self):
        result_layout = QVBoxLayout()
        result_layout.setAlignment(Qt.AlignCenter)

        # 结果图片预览区域
        self.result_label = QLabel("结果图片：未生成")
        self.result_label.setObjectName("result_label")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedSize(400, 400)

        # 进度显示
        self.progress_label = QLabel("进度: 0%")
        self.progress_label.setAlignment(Qt.AlignCenter)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()

        # 控制按钮
        self.start_button = QPushButton("开始风格迁移")
        self.start_button.setIcon(QIcon(resource_path("icons/start.png")))
        
        self.export_button = QPushButton("导出结果")
        self.export_button.setIcon(QIcon(resource_path("icons/export.png")))
        self.export_button.setEnabled(False)

        # 添加到布局
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.progress_label)
        result_layout.addWidget(self.progress_bar)
        result_layout.addWidget(self.start_button)
        result_layout.addWidget(self.export_button)
        
        self.bottom_layout.addLayout(result_layout)

    def setup_connections(self):
        self.content_button.clicked.connect(self.select_content_image)
        self.style_button.clicked.connect(self.select_style_image)
        self.start_button.clicked.connect(self.run_style_transfer)
        self.export_button.clicked.connect(self.export_result)

    def update_progress(self, value):
        """更新进度条和进度标签"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"进度: {value}%")

    def show_message(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def select_content_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择内容图片", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.content_path = path
            self.content_label.setText(f"内容图片：{os.path.basename(path)}")
            self.display_image(path, self.content_image_display)

    def select_style_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择风格图片", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.style_path = path
            self.style_label.setText(f"风格图片：{os.path.basename(path)}")
            self.display_image(path, self.style_image_display)

    def display_image(self, image_path, label):
        """加载并显示图片"""
        pixmap = QPixmap(image_path).scaledToWidth(300, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def run_style_transfer(self):
        if not (self.content_path and self.style_path):
            self.show_message("错误", "请先选择内容图片和风格图片！")
            return

        # 重置并显示进度条
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.progress_label.setText("进度: 0%")
        
        # 禁用相关按钮
        self.start_button.setEnabled(False)
        self.content_button.setEnabled(False)
        self.style_button.setEnabled(False)
        self.result_label.setText("正在处理，请稍候...")

        # 启动后台线程
        self.worker = StyleTransferWorker(self.content_path, self.style_path)
        self.worker.finished.connect(self.on_style_transfer_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.start()

    def on_style_transfer_finished(self, output_path):
        """风格迁移完成后更新界面"""
        # 恢复按钮状态
        self.start_button.setEnabled(True)
        self.content_button.setEnabled(True)
        self.style_button.setEnabled(True)
        self.export_button.setEnabled(True)
        
        # 隐藏进度条并更新进度标签
        self.progress_bar.hide()
        self.progress_label.setText("处理完成！")
        
        # 保存结果路径并显示结果图片
        self.result_path = output_path
        self.result_label.setText("")
        pixmap = QPixmap(output_path).scaledToWidth(400, Qt.SmoothTransformation)
        self.result_label.setPixmap(pixmap)
        
        self.show_message("提示", "风格迁移已完成！")

    def export_result(self):
        if not self.result_path:
            self.show_message("错误", "没有可导出的结果图片！")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果图片", "result.jpg", "Images (*.jpg *.png)"
        )
        if save_path:
            try:
                shutil.copy(self.result_path, save_path)
                self.show_message("提示", f"结果图片已导出到：{save_path}")
            except Exception as e:
                self.show_message("错误", f"导出失败：{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StyleTransferApp()
    window.setWindowIcon(QIcon(resource_path("icons/icon.png")))
    window.show()
    sys.exit(app.exec_())