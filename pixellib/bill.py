import pixellib
from pixellib.instance import instance_segmentation

segment_image = instance_segmentation()
segment_image.load_model("models/mask_rcnn_coco.h5")
segment_image.segmentImage("./../img/img1.png", output_image_name= "output/instance.jpg")