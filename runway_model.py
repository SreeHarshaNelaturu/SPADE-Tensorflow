import runway
from runway.data_types import *
from SPADE import SPADE
from utils import *
import cv2
import tensorflow as tf
import numpy as np


def load_checkpoint(checkpoint, sess):
    saver = tf.train.Saver()
    try:
        saver.restore(sess, checkpoint)
        return True
    except:
        print("checkpoint %s not loaded correctly" % checkpoint)
        return False


@runway.setup(options={"checkpoint": runway.file(is_directory=True)})
def setup(opts):
    global gan
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    gan = SPADE(sess)

    gan.build_model()
    tf.global_variables_initializer().run(session=sess)
    sess = gan.load_model(opts["checkpoint"])

    return sess


label_to_id = {'background': 0, 'hair': 1, 'skin': 2, 'l_brow': 3, 'r_brow': 4, 'l_eye': 5, 'r_eye': 6, 'nose': 7,
               'u_lip': 8, 'l_lip': 9, 'mouth': 10, 'neck': 11, 'r_ear': 12, 'l_ear': 13, 'cloth': 14, 'hat': 15,
               'ear_r': 16, 'eye_g': 17, 'neck_l': 18}


label_to_color = {"background": [0, 0, 0], "hair": [0, 0, 255], "skin": [255, 0, 0], "l_brow": [150, 30, 150],
                  "r_brow": [255, 65, 255], "l_eye": [150, 80, 0], "r_eye": [170, 120, 65], "nose": [125, 125, 125],
                  "u_lip": [255, 255, 0], "l_lip": [0, 255, 255], "mouth": [255, 150, 0], "neck": [255, 225, 120],
                  "r_ear": [255, 125, 125], "l_ear": [200, 100, 100], "cloth": [0, 255, 0], "hat": [0, 150, 80],
                  "ear_r": [215, 175, 125], "eye_g": [220, 180, 210], "neck_l": [125, 125, 255]}


command_inputs = {
    'semantic_map': runway.segmentation(label_to_id=label_to_id, label_to_color=label_to_color,
                                        default_label='background', width=256, height=256),
    'style_image': runway.image
}

command_outputs = {
    'output': runway.image
}


@runway.command("spade_face", inputs=command_inputs, outputs=command_outputs, description="spade_face")
def spade_face(sess, inputs):
    input_seg = load_segmap(np.array(inputs["semantic_map"]), 256, 256, 3)
    guide_img = load_style_image(np.array(inputs["style_image"]), 256, 256, 3)

    fake_img = sess.run(gan.guide_test_fake_x, feed_dict={gan.test_segmap_image: input_seg, gan.test_guide_image: guide_img})

    output = save_images(fake_img, [1,1], "mara_11.jpg")
    print(output)
    return output


if __name__ == "__main__":
    runway.run(model_options={"checkpoint": "./checkpoint"})
