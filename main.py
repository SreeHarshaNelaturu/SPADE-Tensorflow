from SPADE import SPADE
from utils import *
import cv2
import tensorflow as tf
"""main"""
def main():

    checkpoint_dir = "/home/harsha/Documents/SPADE_FACE/og_spade/checkpoint"

    # open session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    gan = SPADE(sess)

    guide_img = cv2.imread("/home/harsha/Documents/SPADE_FACE/og_spade/robin.jpg")
    segmap_in = cv2.imread("./dataset/spade_celebA/segmap_test/408.jpg")
    # build graph
    gan.build_model()
    tf.global_variables_initializer().run(session=sess)


    sess_out = gan.load_model(checkpoint_dir)
    guide_img = load_style_image(guide_img, 256, 256, 3)
    segmap_in = load_segmap(segmap_in, 256, 256, 3)

    fake_img = sess_out.run(gan.guide_test_fake_x, feed_dict={gan.test_segmap_image: segmap_in, gan.test_guide_image: guide_img})
    print(fake_img)
    save_images(fake_img, [1, 1], "./lolwa.jpg")

    print(" [*] Guide test finished")


if __name__ == '__main__':
    main()
