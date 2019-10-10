import runway
from runway.data_types import *
from SPADE import SPADE
from utils import *
from PIL import Image
import tensorflow as tf
import numpy as np
import os

@runway.setup(options={"checkpoint": runway.file(is_directory=True)})
def setup(opts):
    global gan
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    gan = SPADE(sess)

    gan.build_model()
    tf.global_variables_initializer().run(session=sess)
    sess_out = gan.load_model(opts["checkpoint"])

    return sess_out


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


@runway.command("generate_face", inputs=command_inputs, outputs=command_outputs, description="Generates a face using SPADE")
def generate_face(sess_out, inputs):
    original_size = inputs['semantic_map'].size
    segmap_input = np.array(inputs["semantic_map"].resize((256, 256)))
    style_image = load_style_image(np.array(inputs["style_image"]), 256, 256, 3)
    segmap_onehot = np.eye(19)[segmap_input]
    segmap_onehot = np.expand_dims(segmap_onehot, axis=0)
    fake_img = sess_out.run(gan.guide_test_fake_x, feed_dict={gan.test_segmap_image: segmap_onehot,
                                                              gan.test_guide_image: style_image})
    output = save_images(fake_img, [1, 1])
    print(" [*] Guide test finished")
    output = (output * 255).astype(np.uint8)
    output = Image.fromarray(output).resize(original_size)
    return {'output': output}


if __name__ == "__main__":
    runway.run(model_options={"checkpoint": "./checkpoint"})
