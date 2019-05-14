import os, argparse, sys, random, math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
import cv2
import numpy as np

import colorsys
import random

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join("config/coco/"))  # To find local version
from config.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join("pre-trained/mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

"""

1、加载Model
2、预测结果：mask、class、box

3、plt mask box 画上，划到我们需要标注的图片上去

4、实现自动标注和图像识别


"""




def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
       for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
       return image

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,image_mask=True,image_file=None):
    """
        take the image and results and apply the mask, box, and Label
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)

    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
    
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i].decode('utf8')
        ax.text(x1, y1 + 16, caption,
                color='W', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        if image_mask:
            padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
               verts = np.fliplr(verts) - 1
               p = Polygon(verts, facecolor="none", edgecolor=color)
               ax.add_patch(p)
            ax.imshow(masked_image.astype(np.uint8))
            plt.savefig(image_file)
        else:
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            masked_image=apply_mask(image, mask, color)
            masked_image = cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, 2)
            masked_image = cv2.putText(masked_image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
            return masked_image

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', '人', '自行车', '轿车', '摩托车', '飞机',
            '大客车', '火车', '大卡车', '船', '红绿灯',
            '消防栓', '停车标志', '停车计时器', '长凳', '鸟儿',
            '猫咪', '狗', '马', '绵羊', '奶牛', '大象', '熊',
            '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带',
            '行李箱', '飞盘', '滑雪杆', '滑雪板', '球',
            '风筝', '棒球棒', '棒球手套;', '滑板',
            '冲浪板', '网球拍', '瓶子', '玻璃杯', '杯子',
            '餐叉', '小刀', '调羹', '碗', '香蕉', '苹果',
            '三明治', '橘子', '西兰花', '胡萝卜', '热狗', '披萨',
            '甜甜圈', '蛋糕', '椅子', '长沙发', '盆栽', '床',
            '餐桌', '洗手间', '电视', '笔记本电脑', '鼠标', '遥控器',
            '键盘', '手机', '微波炉', '烤箱', '烤面包机',
            '水池', '冰箱', '书', '钟表', '花瓶', '剪刀',
            '玩具熊', '吹风机', '牙刷']

def maskImage(image_file):
    image = skimage.io.imread(image_file)

# Run detection
    results = model.detect([image], verbose=1)

# Visualize results
    r = results[0]
    display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'],image_file=image_file)
    return image_file,results
def maskVideo(frame):
    #capture = cv2.VideoCapture(0)
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    #while True:
        #ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],image_mask=False
        )
        #cv2.imshow('frame', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
        return frame

    #capture.release()
    #cv2.destroyAllWindows()

    #results = model.detect([frame], verbose=0)
    #r = results[0]
    #frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],image_mask=False)
    #cv2.imshow('frame', frame)
    #return frame

