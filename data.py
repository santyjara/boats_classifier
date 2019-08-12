
# coding: utf-8

import os
import sys
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import defaultdict
import xml.etree.ElementTree as ET

def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        file_name = root.find('filename').text
        file_path = root.find('path').text
        name = boxes.find('name').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return file_path,file_name, list_with_all_boxes,name

def find_sources(data_path, file_ext='.xml', shuffle=True, exclude_dirs=None):
    if exclude_dirs == None :
        exclude_dirs = set()
    
    elif isinstance(exclude_dirs,(list,tuple)):
        exclude_dirs = set(exclude_dirs)
    else:
        raise NameError('Invalid exclude dirs')
        
    sources=[]
    
    for dir_name in os.listdir(data_path):
        if dir_name in exclude_dirs:
            continue
        
        for file_name in os.listdir(os.path.join(data_path,dir_name)):
            if file_name.endswith(file_ext):
                sources.append(read_content(os.path.join(data_path,dir_name,file_name)))
    
    if shuffle == True:
        random.shuffle(sources) 
    
    return sources

def prepare_dataset(data_path,new_dir):
    sources = find_sources(data_path, file_ext='.xml', shuffle=True, exclude_dirs=None) 
    path,name, boxes,label = zip(*sources)
    
    unique_labels = set(label)
    metadic = defaultdict(list)

    for i in range(len(path)):
        img = plt.imread(path[i])
        
        for j in boxes[i]:
            new_name = str(hash(name[i]) % ((sys.maxsize + 1) * 2)) + str(sum(j)) + '.jpg'
            metadic['name'].append(new_name)
            metadic['img_path'].append(os.path.join(new_dir,new_name))
            metadic['label'].append(label[i])
            metadic['split'].append('train' if random.random() < 0.7 else 'valid')
    
            img_crop = img[j[1]:j[3],j[0]:j[2]]
            plt.imsave(os.path.join(new_dir,new_name),img_crop)
    
    #  Save the json file
    unique_labels = {k:i for i,k in enumerate(unique_labels)}
    with open('labels_map.json','w') as f:
        json.dump(unique_labels,f)
    # Save de metadata in a csv. file
    metaData = pd.DataFrame(metadic)
    metaData.to_csv('metadata.csv', index=False)  
    # Save a one label metadata to test the code
    metaData_test = metaData.drop_duplicates(['label'], keep='first')    
    metaData_test.to_csv('metadata_test.csv', index=False) 
      
    return metaData
    
def build_sources_from_metadata(metadata,data_dir,mode='train',exclude_labels=None):
    with open('labels_map.json') as json_file:  
        labels_map = json.load(json_file)

    if exclude_labels is None:
        exclude_labels = set()
    elif isinstance(exclude_labels,(list,tuple)):
        exclude_labels = set(exclude_labels)
    else:
        raise "Please insert a valid exclude_label data"

    df = metadata.copy()
    df = df[df['split'] == mode]
    include_mask = df['label'].apply(lambda x : x not in exclude_labels)
    df = df[include_mask]
    df['label'] = df['label'].apply(lambda x : labels_map[x] )
    sources = list(zip(df['img_path'],df['label']))
    
    return sources

def make_dataset(sources,training = False , batch_size=1, num_epochs=1, num_parallel_calls=1, shuffle_buffer_size = None):

    def load(row):
        file_path = row['image']
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img)

        return img, row['label']

    def preprocess_data(image):
        image = tf.image.resize(image,size=(32,32))

        image = image /255
        
        return image

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size * 4
    
    image_paths, labels =  zip(*sources)

    ds = tf.data.Dataset.from_tensor_slices({'image':list(image_paths), 'label':list(labels)})

    if training:
        ds.shuffle(shuffle_buffer_size)

    ds = ds.map(load , num_parallel_calls=num_parallel_calls)
    ds = ds.map(lambda x,y: (preprocess_data(x), y))
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size = 1)

    return ds

if __name__ == '__main__':

    # Set a new name to store the images
    new_dir = 'new_images'
    os.mkdir(new_dir)
    new_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),new_dir)

    # Image folder path 
    data_path = '/home/santiago/Projects/boats_classifier/Dataset_prueba'
    prepare_dataset(data_path,new_dir)