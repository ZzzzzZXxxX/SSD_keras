import classes


def train():
    # 处理数据
    from xml_to_data import XML_preprocessor
    import pickle

    xml = XML_preprocessor("data/label_train/")
    data = xml.data
    label = xml.label
    pickle.dump(data, open('data/train.pkl', 'wb'))
    pickle.dump(label, open('data/label.pkl', 'wb'))
    # 训练

    import ssd_train

    ssd_train.train(len(classes.voc_classes), 50, 3e-4, 'data/train/', 'gpu')  # 类别数 迭代数 学习率 训练集路径 使用cpu||gpu


def test():
    # 测试

    import ssd_test

    ssd_test.test(classes.voc_classes, 'checkpoints/weights.hdf5', 'data/test/', 'gpu',
                  0.6, 'output.csv')  # voc_classes 模型 测试集路径 使用cpu||gpu  置信度 输出路径


import sys

if sys.argv[1] == 'train':
    train()
elif sys.argv[1] == 'test':
    test()
