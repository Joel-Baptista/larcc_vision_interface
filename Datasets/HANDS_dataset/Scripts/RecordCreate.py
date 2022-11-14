from PIL import Image
import numpy as np
import codecs
import skimage.util as sk
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import os
import random
import itertools
from object_detection.utils import dataset_util

''' HOW TO USE THIS SCRIPT:
1) Create a .txt file that contains the TRAINING and TEST datasets. This may be done separately according to the user needs,
   for example by picking the subjects and selecting the data needed for training and test. This script DOES NOT create the
   txt files of the dataset, but users may create their own function to do so by using as example the "shuffle_dataset" function provided.
   Be sure that the first line of the .txt file of each dataset is the headers line! It must be something like:
   path1, path2, class1, class2, ..., classN
2) Check that the filepaths saved into the .txt files of the dataset correspond to the actual filepath in your own computer.
   If not, you can easily change them by using the "Find and Replace" function of any text editor.
3) Call this script by using the following command in the right folder (the folder where this script is):
python RecordCreate.py --training_path=PATH-TO-YOUR-TF-FOLDER/Training.record --test_path=PATH-TO-YOUR-TF-FOLDER/Test.record
4) The script asks some questions to the user. Provide the following:
   - the path of the .txt file of the TRAINING dataset
   - if you choose to create the test dataset too, provide the path of the .txt file of the TEST dataset too
   - choose if you want to create the records and the labelmap or only one of the two
   - select the number of additional channels you have
   - if you choose an option that also creates the labelmap, provide the labelmap save path in full
'''

flags = tf.app.flags
flags.DEFINE_string('training_path', '', 'Path to output TFRecord for TRAINING')
flags.DEFINE_string('test_path', '', 'Path to output TFRecord for TEST')
FLAGS = flags.FLAGS

class color:
    ''' Class to make pretty terminal outputs with colors! '''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def list_create(filename):
    ''' Function needed to load the annotation files and save the whole content into
    a list where each cell corresponds to a single istance (row) of the annotation file. '''

    with codecs.open(filename, "r", encoding="utf-8-sig") as file:
        bbox = file.read().split('\r\n')
    arr = [str(r) for r in bbox]
    if arr[-1] == '':
        # this is necessary if a newline happens to be at the end of the file by mistake
        del arr[-1]
    return arr

def initialize_from_single_files():
    ''' Function needed to initialize the system variables such as the paths of the
    dataset files (both training and test). Please note that the annotation files contain for each row
    pathRGB, pathDepth, Class1, ..., ClassN
    where if no class is detected in the image a placeholder [0 0 0 0] is present, otherwise there are
    the coordinates of the box in MATLAB coordinates. These must be converted afterward. '''

    # loads training dataset
    training_dict = {}
    print(color.BOLD + color.CYAN + 'PLEASE WRITE THE NAME OF THE FILE OF YOUR TRAINING DATASET WITH EXTENSIONS (FOR EXAMPLE /home/ubuntu/training.txt).' + color.END)
    train = raw_input("[TRAINING DATASET PATH]: ")
    training_dict = list_create(train)

    # this creates a dictionary that in dict[0] contains the headers needed to address the classes automatically
    # and in positions 1 to end contains single istances like "path, class1, ..., classN"

    # separates the items according to commas
    training_dict = [str(training_dict[r]).split(',')[:] for r in range(0,len(training_dict))]
    # deletes the brackets
    # please note that in row 0 there are the headers needed to initialize the record
    for i in range(1,len(training_dict)):
        training_dict[i] = [s.strip('[') for s in training_dict[i]]
        training_dict[i] = [s.strip(']') for s in training_dict[i]]

    flag = None
    # if the user wants to create a test dataset too, it must be provided here. Otherwise it only loads the training one
    while flag not in ("y","Y","yes","Yes","YES","n","N","no","No","NO"):
        print(color.BOLD + color.YELLOW + 'DO YOU WANT TO SAVE THE TEST DATASET TOO?' + color.END)
        flag = raw_input("[Y or N]: ")

        if (flag == "y" or flag == "Y" or flag == "yes" or flag == "Yes" or flag == "YES"):
            print(color.BOLD + color.CYAN + 'PLEASE WRITE THE NAME OF THE FILE OF YOUR TEST DATASET WITH EXTENSIONS (FOR EXAMPLE /home/ubuntu/test.txt).' + color.END)
            test = raw_input("[TEST DATASET PATH]: ")
            test_dict = list_create(test)

            test_dict = [str(test_dict[r]).split(',')[:] for r in range(0,len(test_dict))]
            # deletes the brackets
            # please note that in row 0 there are the headers needed to initialize the record
            for i in range(1,len(test_dict)):
                test_dict[i] = [s.strip('[') for s in test_dict[i]]
                test_dict[i] = [s.strip(']') for s in test_dict[i]]

        if (flag == "n" or flag == "N" or flag == "no" or flag == "No" or flag == "NO"):
            test_dict = False
            print(color.BOLD + color.YELLOW + 'OKAY! ONLY TRAINING DATASET PROVIDED.' + color.END)

    print(color.BOLD + color.GREEN + '...DONE!' + color.END)

    return training_dict, test_dict

def initialize_lists():
    ''' Initializes the empy lists for the current record. '''

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    return xmins, xmaxs, ymins, ymaxs, classes_text, classes

def bbox_extract(xmins, xmaxs, ymins, ymaxs, classes_text, classes, bbox_array, bbox_name, bbox_id, height, width):
    ''' Function needed to extract the MATLAB box coordinates and convert them according to the TensorFlow definition of coordinates.
    bbox_array contains the boxes
    bbox_name contains the names corresponding to the box classes
    bbox_id contains the id corresponding to the class'''

    bbox_array = bbox_array.split(' ')[:]
    # List of normalized left x coordinates in bounding box (1 per box)
    xmin = (float(bbox_array[0])) / float(width)
    # List of normalized right x coordinates in bounding box (1 per box)
    xmax = (float(bbox_array[0]) + float(bbox_array[2])) / float(width)
    # List of normalized top y coordinates in bounding box (1 per box)
    ymin = (float(bbox_array[1])) / float(height)
    # List of normalized bottom y coordinates in bounding box (1 per box)
    ymax = (float(bbox_array[1]) + float(bbox_array[3])) / float(height)

    if (xmin<0):
        xmin = 0.0
    if (xmin>1.0):
        xmin = 1.0
    if (xmax<0):
        xmax = 0.0
    if (xmax>1.0):
        xmax = 1.0
    if (ymin<0):
        ymin = 0.0
    if (ymin>1.0):
        ymin = 1.0
    if (ymax<0):
        ymax = 0.0
    if (ymax>1.0):
        ymax = 1.0
    xmins.append(xmin)
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)
    # List of string class name of bounding box (1 per box)
    classes_text.append(bbox_name)
    # List of integer class id of bounding box (1 per box)
    classes.append(int(bbox_id))

def create_tf_record(dictionary, names, ndata):
    ''' Function to create the TF record of a single istance, corresponding to a single row
    of the dataset. It must be called in a loop changing the dictionary item, which must
    contain the current row of the whole dataset to process. '''

    # this is equal to the placeholder for the null value of a gesture
    z = '0 0 0 0'
    # finds the path and at least one box not equal to the null value
    current = [x for x in dictionary if x not in z]
    classID = []
    classNames = []

    # extracts the record with the valid boxes coordinates and the corresponding classes
    for i in range(ndata,len(current)):
        # check if the class index is correct this way by subtracting the channels and adding 1
        classID.append(dictionary.index(current[i])-ndata+1)
        classNames.append(names[dictionary.index(current[i])])

    with tf.gfile.GFile(dictionary[0], 'rb') as fid:
        # reads the image
        encoded_jpg = fid.read()
    if ndata > 1:
        with tf.gfile.GFile(dictionary[1], 'rb') as fid:
            # if more than one type of data it means that there's also depth
            # hence it must be read too and encoded accordingly
            encoded_depth = fid.read()
        inputs_stacked = np.dstack((encoded_jpg, encoded_depth))
        encoded_inputs = inputs_stacked.tostring()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)

    # saves the image dimensions
    height, width = image.size
    # converts to bytes the image format
    format = b'png'

    # converts and encodes the path
    filename = dictionary[0].encode('utf8')

    # initializes the null lists
    [xmins, xmaxs, ymins, ymaxs, classes_text, classes] = initialize_lists()

    # converts the boxes in the right coordinates while creating the correct form of a record
    for k in range(0,len(classNames)):
        # loops on the class names to convert the single box
        # note that we pass k+ndata to current in order to skip the cells containing the file paths
        # which may be more than 1 if different images are to be joined (e. g. rgb image + depth image)
        bbox_extract(xmins, xmaxs, ymins, ymaxs, classes_text, classes, current[k+ndata], classNames[k].encode('utf8'), classID[k], height, width)

    # writes one line of the TF record corresponding to the single istance of the whole dataset dictionary
    # this must be repeated in the main as many times as the length of the dataset!
    if ndata > 1:
        # creates the record using the encoded input with data stacked
        tf_record = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(filename),
          'image/source_id': dataset_util.bytes_feature(filename),
          'image/encoded': dataset_util.bytes_feature(encoded_jpg),
          'image/additional_channels/encoded': dataset_util.bytes_feature(encoded_inputs),
          'image/format': dataset_util.bytes_feature(format),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    else:
        tf_record = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(filename),
          'image/source_id': dataset_util.bytes_feature(filename),
          'image/encoded': dataset_util.bytes_feature(encoded_jpg),
          'image/format': dataset_util.bytes_feature(format),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    return tf_record

def create_labelmap(dictionary, ndata):
    ''' Function to create and save the labelmap needed by TensorFlow. '''

    print(color.BOLD + color.CYAN + 'SAVE PATH OF THE LABELMAP WITH EXTENSION (FOR EXAMPLE /home/ubuntu/labelmap.pbtxt).' + color.END)
    path = raw_input("[PATH]: ")

    print(color.BOLD + color.YELLOW + 'CREATING THE LABELMAP...' + color.END)
    mapfile = open(path,'w')
    for i in range(ndata,len(dictionary[0])):
        mapfile.write('item {\n')
        mapfile.write('  name: \'' + str(dictionary[0][i]) + '\'\n')
        mapfile.write('  id: ' + str(i) + '\n')
        mapfile.write('}\n')
    mapfile.close()
    print(color.BOLD + color.GREEN + 'DONE!' + color.END)

def main(_):
    
    print(color.BOLD + color.YELLOW + '--- STARTING PROGRAM ---' + color.END)
    [training_dict, test_dict] = initialize_from_single_files()
    print(color.BOLD + color.GREEN + 'SUCCESSFULLY INITIALIZED DATASETS!' + color.END)

    print(color.BOLD + color.CYAN + 'WHAT DO YOU WANT TO DO?' + color.END)
    print(color.BOLD + color.CYAN + '0: creates only TF records' + color.END)
    print(color.BOLD + color.CYAN + '1: creates TF records and labelmap' + color.END)
    print(color.BOLD + color.CYAN + '2: creates only labelmap' + color.END)
    flag = raw_input("[Choose 0, 1 or 2]: ")
    flag = int(flag)

    print(color.BOLD + color.CYAN + 'HOW MANY ADDITIONAL CHANNELS DO YOU HAVE?' + color.END)
    ndata = raw_input("[0 means 3 CHANNELS (RGB), 1 means 4 CHANNELS (RGB + D), 2 means 5 CHANNELS, etc.]: ")
    # adds 1 to the number of additional channels to simplify some indexing later
    ndata += 1

    if flag != 0:
        # creates labelmap in cases 1 and 2
        create_labelmap(training_dict,int(ndata))

    if flag != 2:
        # creates TF records in cases 0 and 1
        writer = tf.python_io.TFRecordWriter(FLAGS.training_path)

        for record in range(1,len(training_dict)):
            # starts from 1 because at dict0 we have the headers
            # pass the current record to the function, hence it is a single line that is elaborated
            # second element contains the headers and third element contains the total number of channels
            tf_record = create_tf_record(training_dict[record], training_dict[0], int(ndata))
            # writes the record into the file
            writer.write(tf_record.SerializeToString())
        writer.close()
        training_path = os.path.join(os.getcwd(), FLAGS.training_path)
        print(color.BOLD + color.GREEN + 'SUCCESSFULLY CREATED THE TRAINING TFRECORDS. SAVEPATH: {}'.format(training_path) + color.END)

        if (test_dict != False):
            # also creates the test record
            writer = tf.python_io.TFRecordWriter(FLAGS.test_path)
            for record in range(1,len(test_dict)):
                tf_record = create_tf_record(test_dict[record], test_dict[0], int(ndata))
                # writes the record into the file
                writer.write(tf_record.SerializeToString())
            writer.close()
            test_path = os.path.join(os.getcwd(), FLAGS.test_path)
            print(color.BOLD + color.GREEN + 'SUCCESSFULLY CREATED THE TEST TFRECORDS. SAVEPATH: {}'.format(test_path) + color.END)
            print(color.BOLD + color.YELLOW + 'CLOSING PROGRAM...' + color.END)
        else:
            print(color.BOLD + color.YELLOW + 'ONLY TRAINING DATASET PROVIDED! CLOSING...' + color.END)

if __name__ == '__main__':
  tf.app.run()
