# Headline

> An awesome project.



# 张量（Tensor）

- 0维张量/**标量** 标量是一个数字

- 1维张量/**向量** 1维张量称为”向量“

- 2维张量/**矩阵** 2维张量称为**矩阵**

- 3维张量 公用数据存储在张量

在Python中，张量通常存储在Nunpy数组，Numpy是在大部分的AI框架中，一个使用频率非常高的用于科学计算的数据包。

**我们为什么想把数据转换为Numpy数组？**

> 很简单。因为我们需要把所有的输入数据，如字符串文本，图像，股票价格，或者视频，转变为一个统一得标准，以便能够容易的处理。

- 4维 = 图像 

一张图片有三个参数：**高度、宽度和颜色深度**。

- 5维 = 视频

5D张量可以用来存储视频数据。TensorFlow中，视频数据将如此编码：

（sample_size, frames, width, height, color_depth)

如果我们考察一段5分钟（300秒），1080pHD（1920 x 1080像素），每秒15帧（**总共4500帧**），颜色深度为3的视频，我们可以用4D张量来存储它：

```
(4500,1920,1080,3)
```

当我们有多段视频的时候，张量中的第五个维度将被使用。如果我们有10段这样的视频，我们将得到一个5D张量：

```
(10,4500,1920,1080,3)
```

这个张量的大是很荒谬的，超过1TB。我们姑且考虑下这个例子以便说明一个问题：在现实世界中，我们有时需要尽可能的缩小样本数据以方便的进行处理计算，除非你有无尽的时间。

你不能什么工作也不做就把大堆数据扔向你的AI模型。你必须**清洗和缩减那些数据让后续工作更简洁更高效**。

**降低分辨率**，去掉不必要的数据（也就是去重处理），这大大缩减了帧数，等等这也是数据科学家的工作。如果你不能很好地对数据做这些预处理，那么你几乎做不了任何有意义的事。

**我们将让张量“流动Flow起来”。**



# Python

## 读取文件

```
with open(file, 'rb') as f:
	dic = pickle.load(f, encoding = 'bytes')
data = dic[b'data']
labels = dic[b'labels']

```



## 显示图片

```
#################################PyTorch教程############################################
plt.figure()
for i in range(4):
	ax = plt.subplots(1,4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
	if i == 4:
		plt.show()
		break
##################################机器学习实战###########################################
fig, axs = plt.subplots(nrows = 2, ncols = 2, sharex = False, sharey = False, figsize = (12, 8))
#画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
	#设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

```



### 数据归一化

```
def autoNorm(dataSet):
    #获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals

```



# CIFAR 图片读取

# 上传数据集到云端

colab 和 kaggle服务器都比较慢 

这里使用**mutilCloud**

将oneDrive 和 Google云盘连接起来！

太强了！！ 还可以连百度网盘 ！

太慢了。。。傻逼

## github 保存代码

https://blog.csdn.net/qq_16855077/article/details/90765864

https://www.pianshen.com/article/6046955079/ 这个好点



**…or create a new repository on the command line**



```
echo "# xty" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M master
git remote add origin https://github.com/xty136694120/xty.git
git push -u origin master
                
```

**…or push an existing repository from the command line**



```
git remote add origin https://github.com/xty136694120/xty.git
git branch -M master
git push -u origin master
```

## 保存分支文件

克隆指定的分支：**git clone -b**  **分支名 仓库地址**



# PyTorch 安装

```
conda create -n name python=3.7
conda activate name
```

进入[PyTorch官网](https://pytorch.org/get-started/locally/)选择你的版本

**安装清华镜像源**

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --set show_channel_urls yes

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

**注意**： 后面的 -c pytorch要删掉 不然还是官方的下载源

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2
```

**测试**

```
import torch
print(torch.__version__)
torch.cuda.is_available()
```

输出：True

**删除conda**

```
conda remove -n name --all
```



**删除镜像源**

注：如果要删除自定义源更换回conda的默认源，直接删除channels即可，命令如下：

```
conda config --remove-key channels
```

pip批量导出包含环境中所有组件的requirements.txt文件

```
pip freeze > requirements.txt
```

pip批量安装requirements.txt文件中包含的组件依赖

```
pip install -r requirements.txt
```

conda批量导出包含环境中所有组件的requirements.txt文件

```
conda list -e > requirements.txt
```

conda批量安装requirements.txt文件中包含的组件依赖

```
conda install --yes --file requirements.txt
```



# YOLOv5

##  模型训练相关的参数配置

| 命令行参数 |          参数含义          |        设置值         |
| :--------: | :------------------------: | :-------------------: |
|   --img    |      统一输入图像规模      |          320          |
|  --batch   | 每次网络训练输入图像的数量 |           4           |
|  --epochs  |  整个数据集参与训练的次数  |          100          |
|   --data   |     数据集配置文件路径     |   ./data/taco.yaml    |
|   --cfg    |      模型配置文件路径      | ./models/yolov5s.yaml |
|  --device  |   训练的设备(CPU or GPU)   |          cpu          |
| --weights  |    预训练模型的权重文件    |      yolov5s.pt       |

## 参考文献

https://www.zhihu.com/search?type=content&q=yolo%20%E5%8F%A3%E7%BD%A9%E8%AF%86%E5%88%AB



**YOLOv5从入门到部署之：配置与初始化超参**

https://zhuanlan.zhihu.com/p/296592676

## 数据预处理

1. 本数据来源于MAFA dataset和WIDER FACE dataset。
2. 训练集共6120张图片，来自与MAFA的有3006张图片（基本都是戴口罩的图片），
   WIDER Face的有3114张（基本都是不戴口罩的图片）。
3. 验证集共1839张图片，其中来自MAFA的有1059张图片， 来自于WIDER Face的共780张
4. MAFA数据集链接：http://www.escience.cn/people/geshiming/mafa.html
5. WIDER Face数据集链接：http://shuoyang1213.me/WIDERFACE/

这里由于他将数据分为三类 戴口罩 不戴口罩 没带好口罩 

后期换成别的数据集



- （本次用的就是这个）AIZOO 数据集：https://github.com/AIZOOTech/FaceMaskDetection

- RMFD 数据集 https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset （真实口罩人脸识别数据集：从网络爬取样本，经过整理，清洗和标注后，含525人的5千张口罩人脸，9万正常人脸。）

其中xml 文件里的标签name 分别为```face  face_mask```

需要在convert_annotation修改最上面的classes[] 

这样下面的convert函数就能得到class索引0.1.2..

提取出来返回给txt文件



运行到一半出现错误

```
AttributeError: 'NoneType' object has no attribute 'find'
```

这是由于 **ElementTree find（）始终返回None**

所以size 是NoneType类型

**解决方案**：

- 手动删除 或者在里面手动添加size配置信息
- 好愚蠢 自己手动添加了半个小时 发现不如用if语句判断 然后把分辨率加进去
- 发现新bug size为 0 0 和上步骤一样 重新加size

## 转化好txt文件 

convert_annotation.py

```
import xml.etree.ElementTree as ET
import pickle
import os
from PIL import Image
from os import listdir, getcwd
from os.path import join

classes = ['face_mask','face'] #对应里面的name，按照实际情况修改 这里由于xml文件和.yaml顺序不一 所以调整一下
Annotation_path = 'E:\\project\\yolov5\\dataset\\train\\annotations'
Labels_path = 'E:\\project\\yolov5\\dataset\\train\\labels'
Image_path = 'E:\\project\\yolov5\\dataset\\train\\images'
#这个函数是voc自己的不用修改，在下面的函数中调用
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x,y,w,h)

#生成标签函数，从xml文件中提取有用信息写入txt文件
def convert_annotation(image_id):
    in_file = open(Annotation_path +'\%s.xml'%(image_id)) #Annotations文件夹地址
    out_file = open(Labels_path + '\%s.txt'%(image_id), 'w') #labels文件夹地址
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    print(Image_path +'\%s.jpg'%(image_id))
    if size is None:
        img = Image.open(Image_path + '\%s.jpg' % (image_id))
        w = int(img.width)
        h = int(img.height)
    else:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if w == 0 or h == 0:
            img = Image.open(Image_path + '\%s.jpg' % (image_id))
            w = int(img.width)
            h = int(img.height)



    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls == 'face_nask':
            cls = 'face_mask'
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if not os.path.exists(Labels_path):#不存在文件夹
    os.makedirs(Labels_path)

image_adds = os.listdir( Image_path )

for image_add in image_adds:
    image_add =  image_add.strip().replace('.jpg','')
    convert_annotation(image_add)

print("Finished")

```

发现导出的txt文件小数点超过很多位数 但是案例是取小数点后六位



出现错误

```
Traceback (most recent call last):
  File "train.py", line 487, in <module>
    train(hyp, opt, device, tb_writer, wandb)
  File "train.py", line 58, in train
    data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
  File "D:\Anaconda\envs\yolov5\lib\site-packages\yaml\__init__.py", line 112, in load
    loader = Loader(stream)
  File "D:\Anaconda\envs\yolov5\lib\site-packages\yaml\loader.py", line 24, in __init__
    Reader.__init__(self, stream)
  File "D:\Anaconda\envs\yolov5\lib\site-packages\yaml\reader.py", line 85, in __init__
    self.determine_encoding()
  File "D:\Anaconda\envs\yolov5\lib\site-packages\yaml\reader.py", line 124, in determine_encoding
    self.update_raw()
  File "D:\Anaconda\envs\yolov5\lib\site-packages\yaml\reader.py", line 178, in update_raw
    data = self.stream.read(size)
UnicodeDecodeError: 'gbk' codec can't decode byte 0xad in position 4: illegal multibyte sequence
Internal process exited
```

查的原因是

https://www.cnblogs.com/tina-ma/p/3924854.html

python读取txt文件编码有误

对train.py 下面一行代码 进行修改

```
# Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    # with open(opt.data) as f:
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
```

**同样的test.py这个位置也需要修改**



## 训练模型

```
python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt
```

之前用coco128训练集来测试这个功能时

显示页面大小不足 经查询得到是因为参数权重太大

main函数里面的maximum number of dataloader workers 默认值修改为0

修改后为

```
python train.py --img 320 --batch 8 --epochs 5 --data coco128.yaml --weights yolov5s.pt --workers 0
```

.yaml文件是模型参数文件 yolov5s.pt是模型权重文件 自行修改



## 可视化数据 wandb 

**只需10分钟带你完美入门轻量级可视化工具wandb**
https://zhuanlan.zhihu.com/p/266337608

wandb API:   6a395aec190e8b43e0dd6de11b7de330f6bc5101

报错 https://github.com/wandb/client/issues/1236

解决方案 更改wandb版本

pip install wandb==0.9.7

## 保存模型

https://blog.csdn.net/dzg_chat/article/details/89483321

```
# by default, this will save to a new subfolder for files associated
# with your run, created in wandb.run.dir (which is ./wandb by default)
wandb.save("mymodel.h5")

# you can pass the full path to the Keras model API
model.save(os.path.join(wandb.run.dir, "mymodel.h5"))
```

