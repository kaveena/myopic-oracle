#!/usr/bin/env python3
import copy
import os
from subprocess import call
import numpy as np
import sklearn.cross_validation
import pickle
import cv2
import shutil

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin-1')
    fo.close()
    return dict

def shuffle_data(data, labels):
    data, _, labels, _ = sklearn.cross_validation.train_test_split(data, labels, test_size=0.0, random_state=42)
    return data, labels

def load_data(train_batches):
    data = []
    labels = []
    for data_batch_i in train_batches:
        d = unpickle(
            os.path.join(cifar_python_directory, data_batch_i)
        )
        data.append(d['data'])
        labels.append(np.array(d['labels']))
    # Merge training batches on their first dimension
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    length = len(labels)

    data, labels = shuffle_data(data, labels)

    return data.reshape(length, 3, 32, 32), labels

def load_label_names():
  d = unpickle (
      os.path.join(cifar_python_directory, "batches.meta")
  )
  return d

if __name__ == "__main__":

  cifar_python_archive = os.path.abspath("cifar-10-python.tar.gz")

  if not os.path.exists(cifar_python_archive):
    print("Downloading CIFAR10...")
    call(
      "wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
      shell=True
    )
    print("Downloading done.\n")

  cifar_python_directory = os.path.abspath("cifar-10-batches-py")

  print("Extracting...")

  call(
    "tar -zxvf cifar-10-python.tar.gz",
    shell=True
  )

  print("Extracting successfully done to {}".format(cifar_python_directory))

  print("Converting...")

  cifar_caffe_directory = os.path.abspath('./data')

  X, y = load_data(
      ["data_batch_{}".format(i) for i in range(1, 6)]
  )

  Xt, yt = load_data(["test_batch"])

  L = load_label_names()

  if not os.path.exists(cifar_caffe_directory):
    os.makedirs(cifar_caffe_directory)

  train_index = 0
  train_listing = ""
  for (img, label) in zip([x for x in X], [l for l in y]):
    fname = "train-{}.png".format(train_index)
    r = img[0].astype(np.uint8)
    g = img[1].astype(np.uint8)
    b = img[2].astype(np.uint8)
    output = cv2.merge((b, g, r))
    cv2.imwrite(os.path.join(cifar_caffe_directory, fname), output)
    train_listing += cifar_caffe_directory + "/" + fname + " " + str(label) + "\n"
    train_index += 1

  with open(os.path.join(cifar_caffe_directory, "train-index.txt"), "w") as text_file:
    text_file.write(train_listing)

  test_index = 0
  test_listing = ""
  for (img, label) in zip([x for x in Xt], [l for l in yt]):
    fname = "test-{}.png".format(test_index)
    r = img[0].astype(np.uint8)
    g = img[1].astype(np.uint8)
    b = img[2].astype(np.uint8)
    output = cv2.merge((b, g, r))
    cv2.imwrite(os.path.join(cifar_caffe_directory, fname), output)
    test_listing += cifar_caffe_directory + "/" + fname + " " + str(label) + "\n"
    test_index += 1

  with open(os.path.join(cifar_caffe_directory, "test-index.txt"), "w") as text_file:
    text_file.write(test_listing)

  with open(os.path.join(cifar_caffe_directory, "labels.txt"), "w") as text_file:
    text_file.write("\n".join(L['label_names']))

  print("Cleaning up...")

  shutil.rmtree(cifar_python_directory)

  print("Done")
