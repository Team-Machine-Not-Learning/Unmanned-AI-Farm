# YOLOV10训练目标检测模型

python版本3.9
建议用anaconda建一个虚拟环境，在虚拟环境中安装下面的库【注意：一定要先进入虚拟环境再安装库】
建议用pycharm打开项目，并在pycharm中用建好的虚拟环境来运行项目文件
关于用anaconda建虚拟环境和在pycharm中用虚拟环境运行项目，可自行查找

（一）安装torch（有电脑有GPU，则安装GPU版本torch，否则安装CPU版本torch）
    安装CPU版本torch
    ```
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
    ```
    安装GPU版本torch（如果GPU版无法安装成功，可直接安装CPU版的）
    ```
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    ```
（二） 安装其他依赖库（需要具体指定requirements.txt的位置【此文件就在此项目中】，例如：pip install -r C:\Users\2023\Desktop\requirements.txt)
    ```
    pip install -r requirements.txt
    ```
（三）模型训练和验证 【原项目模型已经训练好，可跳过此步骤】
     1) 打开data/data.yaml, 修改train和val的路径为自己数据集的路径
     2) 运行train.py文件，训练模型
     3) 修改val.py文件中模型权重路径，运行val.py文件，验证模型效果

（四）模型应用
    打开gui_chinese.py文件，修改此文件的类class Controller中的model_path模型权重路径，并运行此文件即可使用训练好的模型

（五）注意事项
    如果运行gui_chinese.py文件弹出的界面出现中文乱码，可运行英文版的界面gui_english.py