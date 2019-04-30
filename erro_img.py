import os
from xml.etree import ElementTree


class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path

        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        print(filenames)
        for filename in filenames:

            if filename == '.DS_Store':
                os.remove(self.path_prefix + filename)
                print("remove",self.path_prefix + filename)
                continue
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            if width<300 or height<300:
                os.remove(self.path_prefix + filename)
                print("remove", self.path_prefix + filename)
                continue
# ## example on how to use it
# import pickle
#
#
data = XML_preprocessor("data/label_train/").data
# pickle.dump(data, open('data/train.pkl', 'wb'))
a= dict()
a.values()