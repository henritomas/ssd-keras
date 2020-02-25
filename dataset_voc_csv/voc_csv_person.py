#Original from https://github.com/qqwweee/keras-yolo3/blob/master/voc_annotation.py

"""
Insert VOCdevkit in the same directory as this python script
"""

import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'trainval'), ('2007', 'test')]

classes = ["background", "person",]


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
            
        list_file.write('%s.jpg'%(image_id))
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(","+",".join([str(a) for a in b]) + ',' + str(cls_id)) #Writes class id (e.g. dog= 0) instead of name
        #list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls)) #Writes name (dog) instead of class id
        list_file.write('\n')
wd = getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        #list_file.write('%s.jpg'%(image_id))
        convert_annotation(year, image_id, list_file)
        #list_file.write('\n')
    list_file.close()