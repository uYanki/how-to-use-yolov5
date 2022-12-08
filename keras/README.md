

# 2022.11.29

ASUS ZenBook（MX150）+ Win10

[支持情况][https://www.nvidia.cn/geforce/gaming-laptops/geforce-mx150/]

![image-20221129153259146](Images/image-20221129153259146.png)

### Environment

#### Driver![image-20221129030327824](Images/image-20221129030327824-1669714497547.png)

默认安装适合自己的显卡驱动即可。

#### CUDA

##### Install

1. CUDA version

安装驱动后查看最高支持的CUDA版本。

![image-20221129174410424](Images/image-20221129174410424.png)

2. [tensorflow-gpu][https://tensorflow.google.cn/install/source_windows#gpu] version

![image-20221129174118494](Images/image-20221129174118494.png)

根据 Python 版本选择 tensorflow-gpu 版本、cuDNN、CUDA 版本。

因我电脑安装的是 Python 3.10，所以我选择安装 tensorflow-gpu-2.11，cuDNN-8.1，CUDA-11.2。(注：对应的 python 版本才能找到对应的 tensorflow-gpu 库)

![image-20221129185406161](Images/image-20221129185406161.png)

2. uninstall CUDA

卸载旧的：

![image-20221129032250734](Images/image-20221129032250734.png)

3. install [CUDA][https://developer.nvidia.com/cuda-toolkit-archive]

![image-20221129185704376](Images/image-20221129185704376.png)

![image-20221129185818107](Images/image-20221129185818107.png)

安装 - 自定义

![image-20221129190121627](Images/image-20221129190121627.png)

![image-20221129190008974](Images/image-20221129190008974.png)

![image-20221129185955668](Images/image-20221129185955668.png)

##### Test

获取版本号：`nvcc -V` / `nvcc -version`

![image-20221129222724986](Images/image-20221129222724986.png)

##### Env

不需要编辑环境变量，只是查看而已。

![image-20221129191557995](Images/image-20221129191557995.png)

#### [cuDNN][https://developer.nvidia.com/rdp/cudnn-archive]

##### Install

![image-20221129190550989](Images/image-20221129190550989.png)

![image-20221129190635361](Images/image-20221129190635361.png)

点击下载，得到压缩包，并解压到对应文件夹里。

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
```

![image-20221129190920908](Images/image-20221129190920908.png)

##### Test

查看显卡信息：`nvidia-smi`

![image-20221129192417739](Images/image-20221129192417739.png)

##### Test

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\demo_suite
```

```
bandwidthTest.exe
deviceQuery.exe
```

![image-20221129191912193](Images/image-20221129191912193.png)

![image-20221129192002733](Images/image-20221129192002733.png)

#### TensorFlow

##### Install

```shell
pip install tensorflow-gpu==2.11
```

##### Test

测试 GPU 是否可用：

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

![image-20221129191231990](Images/image-20221129191231990.png)

若出现找不到 dll 的情况，将 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin` 里的文件复制到 `C:\Windows\System32` 里。

GPU 使用情况：

![image-20221129192704549](Images/image-20221129192704549.png)

##### Test

```python
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
```

![image-20221129192256890](Images/image-20221129192256890.png)



