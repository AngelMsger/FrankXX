#!/usr/bin/python
import argparse
import glob
import logging
import os
from shutil import copyfile

import dlib

parser = argparse.ArgumentParser()
parser.add_argument('faces_folder_path', help='Images Source Dir')
parser.add_argument('output_folder_path', help='Clusters Output Dir')
parser.add_argument('--predictor_path', default='shape_predictor_5_face_landmarks.dat',
                    help='Predictor Model Path')
parser.add_argument('--face_rec_model_path', default='dlib_face_recognition_resnet_model_v1.dat',
                    help='Face Rec Model_Path')
parser.add_argument('--cluster_size_threshold', type=int, default=3, help='Cluster Size Threshold')
parser.add_argument('--save_face_chip', type=bool, default=True, help='If Save Face Chip')
args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(args.predictor_path)
facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

descriptors = []
images = []

logger = logging.getLogger('FrankXX')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

for f in glob.glob(os.path.join(args.faces_folder_path, "*.jpg")):
    logger.info("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    faces = detector(img, 1)
    logger.info("Number of faces detected: {}".format(len(faces)))

    for face in faces:
        shape = sp(img, face)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape, f))

labels = dlib.chinese_whispers_clustering(descriptors, 0.5)

num_classes = len(set(labels))
logger.info("Number of clusters: {}".format(num_classes))

clusters = [[] for _ in range(num_classes)]
for i, pair in enumerate(images):
    clusters[labels[i]].append(pair)


def copy(option):
    copyfile(option['f'], option['file_path'])


def save_face_chip(option):
    dlib.save_face_chip(option['img'], option['shape'], option['file_path'], size=150, padding=0.25)


if args.save_face_chip:
    process = save_face_chip
else:
    process = copy

logger.info("Saving faces in largest cluster to output folder...")
for i, cluster in enumerate(clusters):
    if len(cluster) > args.cluster_size_threshold:
        cluster_folder_path = os.path.join(args.output_folder_path, str(labels[i]))
        if not os.path.isdir(cluster_folder_path):
            os.makedirs(cluster_folder_path)
        for j, pair in enumerate(cluster):
            img, shape, f = pair
            process({
                'img': img,
                'shape': shape,
                'file_path': os.path.join(cluster_folder_path, 'face_{}'.format(j)),
                'f': f
            })
