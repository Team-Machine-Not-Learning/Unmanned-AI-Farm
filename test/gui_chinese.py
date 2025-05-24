import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
import torch
import threading
from pathlib import Path
import os
from ultralytics import YOLOv10
import sys
import cv2
import numpy as np
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk  # 位置不同可能报错


warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.filterwarnings("ignore")


class WinGUI(Tk):
    """界面布局类"""

    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_button_upload_image = self.__tk_button_upload_image(self)
        self.tk_button_video = self.__tk_button_video(self)
        self.tk_button_cam = self.__tk_button_cam(self)
        self.tk_label_show_image = self.__tk_label_show_image(self)
        self.tk_button_close = self.__tk_button_close(self)
        self.tk_label_time = self.__tk_label_time(self)
        self.tk_label_time_show = self.__tk_label_time_show(self)
        self.tk_label_count = self.__tk_label_count(self)
        self.tk_label_count_show = self.__tk_label_count_show(self)
        self.tk_label_conf = self.__tk_label_conf(self)
        self.tk_label_conf_show = self.__tk_label_conf_show(self)
        self.tk_label_class = self.__tk_label_class(self)
        self.tk_label_class_show = self.__tk_label_class_show(self)
        self.tk_label_xmin = self.__tk_label_xmin(self)
        self.tk_label_xmin_show = self.__tk_label_xmin_show(self)
        self.tk_label_ymin = self.__tk_label_ymin(self)
        self.tk_label_ymin_show = self.__tk_label_ymin_show(self)
        self.tk_label_xmax = self.__tk_label_xmax(self)
        self.tk_label_xmax_show = self.__tk_label_xmax_show(self)
        self.tk_label_ymax = self.__tk_label_ymax(self)
        self.tk_label_ymax_show = self.__tk_label_ymax_show(self)
        self.tk_label_select_obj = self.__tk_label_select_obj(self)
        self.tk_select_box_select_obj_show = self.__tk_select_box_select_obj_show(self)

    def __win(self):
        self.title("目标检测系统")
        # 设置窗口大小、居中
        width = 519
        height = 619
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)

        self.resizable(width=False, height=False)

    def scrollbar_autohide(self, vbar, hbar, widget):
        """自动隐藏滚动条"""

        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)

        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)

        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())

    def v_scrollbar(self, vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')

    def h_scrollbar(self, hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')

    def create_bar(self, master, widget, is_vbar, is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def __tk_button_upload_image(self, parent):
        btn = Button(parent, text="上传图片", takefocus=False, )
        btn.place(x=62, y=29, width=79, height=30)
        return btn

    def __tk_button_video(self, parent):
        btn = Button(parent, text="上传视频", takefocus=False, )
        btn.place(x=155, y=29, width=84, height=30)
        return btn

    def __tk_button_cam(self, parent):
        btn = Button(parent, text="打开摄像头", takefocus=False, )
        btn.place(x=253, y=29, width=90, height=30)
        return btn

    def __tk_label_show_image(self, parent):
        image = Image.open("images/pre_show.png")
        image = image.resize((400, 400))
        photo = ImageTk.PhotoImage(image)
        label = Label(parent, text="标签", anchor="center", )
        label.place(x=68, y=73, width=400, height=400)
        label.configure(image=photo)
        label.image = photo
        return label

    def __tk_button_close(self, parent):
        btn = Button(parent, text="关闭视频或摄像", takefocus=False, )
        btn.place(x=364, y=29, width=112, height=30)
        return btn

    def __tk_label_time(self, parent):
        label = Label(parent, text="用时：", anchor="center", )
        label.place(x=43, y=488, width=50, height=30)
        return label

    def __tk_label_time_show(self, parent):
        label = Label(parent, text=" ", anchor="center", )
        label.place(x=85, y=488, width=50, height=30)
        return label

    def __tk_label_count(self, parent):
        label = Label(parent, text="目标数目：", anchor="center", )
        label.place(x=182, y=488, width=88, height=30)
        return label

    def __tk_label_count_show(self, parent):
        label = Label(parent, text=" ", anchor="center", )
        label.place(x=254, y=488, width=30, height=30)
        return label

    def __tk_label_conf(self, parent):
        label = Label(parent, text="置信度：", anchor="center", )
        label.place(x=41, y=515, width=50, height=30)
        return label

    def __tk_label_conf_show(self, parent):
        label = Label(parent, text=" ", anchor="center", )
        label.place(x=83, y=515, width=64, height=30)
        return label

    def __tk_label_class(self, parent):
        label = Label(parent, text="类别：", anchor="center", )
        label.place(x=197, y=515, width=50, height=30)
        return label

    def __tk_label_class_show(self, parent):
        label = Label(parent, text=" ", anchor="center", )
        label.place(x=243, y=515, width=119, height=30)
        return label

    def __tk_label_xmin(self, parent):
        label = Label(parent, text="xmin:", anchor="center", )
        label.place(x=41, y=550, width=50, height=30)
        return label

    def __tk_label_xmin_show(self, parent):
        label = Label(parent, text=" ", anchor="center", )
        label.place(x=81, y=550, width=50, height=30)
        return label

    def __tk_label_ymin(self, parent):
        label = Label(parent, text="ymin:", anchor="center", )
        label.place(x=146, y=550, width=50, height=30)
        return label

    def __tk_label_ymin_show(self, parent):
        label = Label(parent, text=" ", anchor="center", )
        label.place(x=186, y=550, width=50, height=30)
        return label

    def __tk_label_xmax(self, parent):
        label = Label(parent, text="xmax:", anchor="center", )
        label.place(x=271, y=550, width=50, height=30)
        return label

    def __tk_label_xmax_show(self, parent):
        label = Label(parent, text=" ", anchor="center", )
        label.place(x=316, y=550, width=50, height=30)
        return label

    def __tk_label_ymax(self, parent):
        label = Label(parent, text="ymax:", anchor="center", )
        label.place(x=385, y=550, width=50, height=30)
        return label

    def __tk_label_ymax_show(self, parent):
        label = Label(parent, text=" ", anchor="center", )
        label.place(x=435, y=550, width=50, height=30)
        return label

    def __tk_label_select_obj(self, parent):
        label = Label(parent, text="目标选择：", anchor="center", )
        label.place(x=384, y=488, width=88, height=30)
        return label

    def __tk_select_box_select_obj_show(self, parent):
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("")
        cb.place(x=382, y=515, width=95, height=30)
        return cb


class Win(WinGUI):
    """界面事件绑定类"""
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)

    def __event_bind(self):
        self.tk_button_upload_image.bind('<Button-1>', self.ctl.upload_image)
        self.tk_button_video.bind('<Button-1>', self.ctl.upload_video)
        self.tk_button_cam.bind('<Button-1>', self.ctl.open_cam)
        self.tk_button_close.bind('<Button-1>', self.ctl.close_video_cam)
        self.tk_select_box_select_obj_show.bind('<<ComboboxSelected>>', self.ctl.combox_change)
        pass

    def __style_config(self):
        pass


class Controller:
    """逻辑处理类，图片检测、视频检测、摄像头检测的逻辑处理"""
    ui: Win

    def __init__(self):
        # todo 修改模型权重路径
        model_path = "runs/detect/train/weights/best.pt"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.stop = False
        self.file_path = ""
        # 图片读取进程
        self.output_size = 640
        self.img2predict = ""
        # 更新视频图像
        self.cap = None
        self.is_camera_open = False
        self.stopEvent = threading.Event()
        self.frame = None
        self.is_playing = False
        self.is_capturing = False
        # 加载检测模型
        self.model = YOLOv10(model_path)
        self.conf_threshold = 0.25
        self.delay = 15  # 视频帧延迟时间

    def init(self, ui):
        """
        得到UI实例，对组件进行初始化配置
        """
        self.ui = ui
        # TODO 组件初始化 赋值操作
        self.ui.tk_button_close.config(state=DISABLED)

    def display_image(self, file_path):
        """显示图片"""
        # 打开图片并调整大小
        image = Image.open(file_path)
        image = image.resize((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.ui.tk_label_show_image.config(image=photo)
        self.ui.tk_label_show_image.image = photo

    def upload_image(self, evt):
        """图像目标检测"""
        self.ui.tk_button_close.config(state=DISABLED)
        self.ui.tk_select_box_select_obj_show.delete(0, tk.END)
        self.ui.tk_select_box_select_obj_show.config(state=tk.NORMAL)
        # 打开文件对话框选择图片文件
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if self.image_path:
            """检测图片"""
            org_path = self.image_path
            # 目标检测
            t1 = time.time()
            results = self.model.predict(source=org_path, imgsz=self.output_size, conf=self.conf_threshold)[0]
            names = results.names
            t2 = time.time()
            self.ui.tk_label_time_show.config(text='{:.3f} s'.format(t2 - t1))
            now_img = results.plot()
            resize_scale = self.output_size / now_img.shape[0]
            im0 = cv2.resize(now_img, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/single_result.jpg", im0)
            self.display_image("images/tmp/single_result.jpg")

            location_list = results.boxes.xyxy.tolist()
            location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            cls_list = [int(i) for i in cls_list]
            conf_list = results.boxes.conf.tolist()
            conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

            total_nums = len(location_list)
            self.ui.tk_label_count_show.config(text="{}".format(total_nums))

            choose_list = ['全部']
            target_names = [names[id] + '_' + str(index) for index, id in enumerate(cls_list)]
            choose_list = choose_list + target_names
            self.ui.tk_select_box_select_obj_show.config(values=choose_list)

            self.results = results
            self.names = names
            self.cls_list = cls_list
            self.conf_list = conf_list
            self.location_list = location_list

            if total_nums >= 1:
                self.ui.tk_label_class_show.config(text="{}".format(names[self.cls_list[0]]))
                self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
                #   设置坐标位置值
                #   默认显示第一个目标框坐标
                self.ui.tk_label_xmin_show.config(text="{}".format(self.location_list[0][0]))
                self.ui.tk_label_ymin_show.config(text="{}".format(self.location_list[0][1]))
                self.ui.tk_label_xmax_show.config(text="{}".format(self.location_list[0][2]))
                self.ui.tk_label_ymax_show.config(text="{}".format(self.location_list[0][3]))
            else:
                self.ui.tk_label_class_show.config(text=" ")
                self.ui.tk_label_conf_show.config(text=" ")
                self.ui.tk_label_xmin_show.config(text=" ")
                self.ui.tk_label_ymin_show.config(text=" ")
                self.ui.tk_label_xmax_show.config(text=" ")
                self.ui.tk_label_ymax_show.config(text=" ")

    def combox_change(self, evt):
        com_text = self.ui.tk_select_box_select_obj_show.get()
        if com_text == '全部':
            cur_box = self.location_list
            cur_img = self.results.plot()
            self.ui.tk_label_class_show.config(text="{}".format(self.names[self.cls_list[0]]))
            self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            cur_img = self.results[index].plot()
            self.ui.tk_label_class_show.config(text="{}".format(self.names[self.cls_list[index]]))
            self.ui.tk_label_conf_show.config(text="{}".format(str(self.conf_list[index])))

        # 设置坐标位置值
        self.ui.tk_label_xmin_show.config(text="{}".format(str(cur_box[0][0])))
        self.ui.tk_label_ymin_show.config(text="{}".format(str(cur_box[0][1])))
        self.ui.tk_label_xmax_show.config(text="{}".format(str(cur_box[0][2])))
        self.ui.tk_label_ymax_show.config(text="{}".format(str(cur_box[0][3])))

        cv2.imwrite("images/tmp/single_result.jpg", cur_img)
        self.display_image("images/tmp/single_result.jpg")

    def upload_video(self, evt):
        """视频目标检测"""
        self.ui.tk_button_upload_image.config(state=tk.DISABLED)
        self.ui.tk_button_video.config(state=tk.DISABLED)
        self.ui.tk_button_cam.config(state=tk.DISABLED)
        self.ui.tk_button_close.config(state=tk.NORMAL)
        self.ui.tk_select_box_select_obj_show.delete(0, tk.END)
        self.ui.tk_select_box_select_obj_show.config(state=tk.DISABLED)
        """开启视频文件检测事件"""
        self.video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.is_playing = True
            self.update_frame()

    def update_frame(self):
        if self.is_playing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 目标检测
                t1 = time.time()
                results = self.model.predict(source=frame, imgsz=self.output_size, conf=self.conf_threshold)[0]
                names = results.names
                t2 = time.time()
                self.ui.tk_label_time_show.config(text='{:.3f} s'.format(t2 - t1))
                now_img = results.plot()
                cv2.imwrite("images/tmp/single_result_vid.jpg", now_img)
                self.display_image("images/tmp/single_result_vid.jpg")

                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                conf_list = results.boxes.conf.tolist()
                conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

                total_nums = len(location_list)
                self.ui.tk_label_count_show.config(text="{}".format(total_nums))

                self.results = results
                self.names = names
                self.cls_list = cls_list
                self.conf_list = conf_list
                self.location_list = location_list

                if total_nums >= 1:
                    self.ui.tk_label_class_show.config(text="{}".format(names[self.cls_list[0]]))
                    self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
                    #   设置坐标位置值
                    #   默认显示第一个目标框坐标，以左上角为原点，向下y轴，向右x轴，min是左上角点坐标，max是右下角点坐标
                    self.ui.tk_label_xmin_show.config(text="{}".format(self.location_list[0][0]))
                    self.ui.tk_label_ymin_show.config(text="{}".format(self.location_list[0][1]))
                    self.ui.tk_label_xmax_show.config(text="{}".format(self.location_list[0][2]))
                    self.ui.tk_label_ymax_show.config(text="{}".format(self.location_list[0][3]))
                else:
                    self.ui.tk_label_class_show.config(text=" ")
                    self.ui.tk_label_conf_show.config(text=" ")
                    self.ui.tk_label_xmin_show.config(text=" ")
                    self.ui.tk_label_ymin_show.config(text=" ")
                    self.ui.tk_label_xmax_show.config(text=" ")
                    self.ui.tk_label_ymax_show.config(text=" ")
            else:
                self.is_playing = False
                self.cap.release()
        if self.is_playing:
            self.ui.after(self.delay, self.update_frame)

    def open_cam(self, evt):
        """打开摄像头目标检测"""
        self.ui.tk_button_upload_image.config(state=tk.DISABLED)
        self.ui.tk_button_video.config(state=tk.DISABLED)
        self.ui.tk_button_cam.config(state=tk.DISABLED)
        self.ui.tk_button_close.config(state=tk.NORMAL)
        self.ui.tk_select_box_select_obj_show.delete(0, tk.END)
        self.ui.tk_select_box_select_obj_show.config(state=tk.DISABLED)
        """开启摄像头检测事件"""
        if not self.is_capturing:
            self.is_capturing = True
            self.cap = cv2.VideoCapture(1)
            #这里选摄像头
            self.update_camera_frame()

    def update_camera_frame(self):
        if self.is_capturing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 目标检测
                t1 = time.time()
                results = self.model.predict(source=frame, imgsz=self.output_size, conf=self.conf_threshold)[0]
                names = results.names
                t2 = time.time()
                self.ui.tk_label_time_show.config(text='{:.3f} s'.format(t2 - t1))
                now_img = results.plot()
                cv2.imwrite("images/tmp/single_result_vid.jpg", now_img)
                self.display_image("images/tmp/single_result_vid.jpg")
                self.ui.after(30, self.update_camera_frame)

                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                conf_list = results.boxes.conf.tolist()
                conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

                total_nums = len(location_list)
                self.ui.tk_label_count_show.config(text="{}".format(total_nums))

                self.results = results
                self.names = names
                self.cls_list = cls_list
                self.conf_list = conf_list
                self.location_list = location_list

                if total_nums >= 1:
                    self.ui.tk_label_class_show.config(text="{}".format(names[self.cls_list[0]]))
                    self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
                    #   设置坐标位置值
                    #   默认显示第一个目标框坐标
                    self.ui.tk_label_xmin_show.config(text="{}".format(self.location_list[0][0]))
                    self.ui.tk_label_ymin_show.config(text="{}".format(self.location_list[0][1]))
                    self.ui.tk_label_xmax_show.config(text="{}".format(self.location_list[0][2]))
                    self.ui.tk_label_ymax_show.config(text="{}".format(self.location_list[0][3]))
                else:
                    self.ui.tk_label_class_show.config(text=" ")
                    self.ui.tk_label_conf_show.config(text=" ")
                    self.ui.tk_label_xmin_show.config(text=" ")
                    self.ui.tk_label_ymin_show.config(text=" ")
                    self.ui.tk_label_xmax_show.config(text=" ")
                    self.ui.tk_label_ymax_show.config(text=" ")
            else:
                self.close_video_cam(None)

    def close_video_cam(self, evt):
        """关闭摄像头和视屏检测事件"""
        self.ui.tk_button_upload_image.config(state=tk.NORMAL)
        self.ui.tk_button_video.config(state=tk.NORMAL)
        self.ui.tk_button_cam.config(state=tk.NORMAL)
        self.ui.tk_button_close.config(state=tk.DISABLED)
        """窗口关闭事件"""
        if self.is_playing or self.is_capturing:
            self.cap.release()
            self.is_playing = False
            self.is_capturing = False


if __name__ == "__main__":
    controller = Controller()
    app = Win(controller)
    app.mainloop()