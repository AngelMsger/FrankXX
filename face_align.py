import os

import dlib

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')


def face_align(src, dest):
    for label in os.listdir(src):
        label_src = os.path.join(src, label)
        label_dest = os.path.join(dest, label)
        if not os.path.isdir(label_dest):
            os.makedirs(label_dest)
        for file in os.listdir(label_src):
            in_abspath = os.path.join(label_src, file)
            img = dlib.load_rgb_image(in_abspath)
            faces = detector(img, 1)
            for i, face in enumerate(faces):
                shape = sp(img, face)
                out_abspath = os.path.join(label_dest, str(i))
                dlib.save_face_chip(img, shape, out_abspath, size=96)


face_align('lfw-deepfunneled', 'lfw-deepfunneled-aligned')

