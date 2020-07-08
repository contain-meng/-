####配置
python 3.6.10

pip 10.0.1

setuptools 39.1.0

keras 2.2.0

tensorflow-gpu 1.10.0

opencv-python 3.3.0.10

Cuda 9.0

Cudnn 7.0.5

####运行

安装依赖库后，运行grt_pic.py文件会调取笔记本摄像头获取人脸图片，
然后运行predict.py文件会进行识别，最终输出各个表情概率

####建议
推荐使用anaconda创建一个gpu环境，将依赖库安装到gpu环境中