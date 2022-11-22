## 本地进行预测

安装 yolov5：

```shell
pip install yolov5
```

将模型文件和测试集复制到 `yolov5-master` 的目录下，执行以下命令：

```shell
python detect.py --weight best.pt --source datatest
```

预测到 `run` 目录里查看预测结果即可。