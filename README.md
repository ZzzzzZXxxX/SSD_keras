# 【目标识别】 SSD_keras训练自己数据集
### 依赖
```
cv2==3.3.0
keras==2.2.0
matplotlib==2.1.0
tensorflow==1.3.0
numpy==1.13.3
```
### step1
git本项目至本地<br/>
下载预训练权重 至data文件夹下<这里提供百度网盘地址>
```
链接:https://pan.baidu.com/s/1Yub2c2WgeHkuV7akTXhKlw  密码:ohkw
```
### step2
#### 训练
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
### step3
- 测试，将测试图片放置于./data/test/ ，执行
```
python main.py test
```





