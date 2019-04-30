#【目标识别】 SSD_keras训练自己数据集
### 依赖
```
cv2==3.3.0
keras==2.2.0
matplotlib==2.1.0
tensorflow==1.3.0
numpy==1.13.3
```
###step1
git本项目至本地<br/>
下载预训练权重 至data文件夹下<这里提供百度网盘地址>
```
链接:https://pan.baidu.com/s/1Yub2c2WgeHkuV7akTXhKlw  密码:ohkw
```
###step2
####训练
#####
- 使用labelImg对数据进行标记
```
https://github.com/tzutalin/labelImg
```
得到xml文件，放置于./data/label_train/ 
将图片数据放在于./data/train/ 
- 执行erro_img.py,将标记数据中，宽高少于300的xml去掉
- 将数据类别写入classes.py的list中
- 开始训练（打开 main.py，修改相关参数，如使用cpu，gpu，迭代数等）,执行
```
python main.py train
```
<如果，训练时报错，可能训练图片不满足SSD_keras的要求，删除修改相关数据>
###step3
- 测试，将测试图片放置于./data/test/ ，执行
```
python main.py test
```

##相关数据集提供
- 旗帜（包含40个种类旗帜），数据来着于网络,数据标注是个苦力活，本数据包含1600多张图片，花费接近一个星期标注完成，且用且珍惜！！！
```

```
在本Demo上，使用本数据集，对于40个类别，存在不收敛现象，原因未知，期待你们的解答。而使用前6类或者前几类实验时却未出现不收敛现象，且测试结果还不错。
因为这个客观问题存在且本人无力解决，故本人转战yolov3_keras对此全部类别识别的实现。




