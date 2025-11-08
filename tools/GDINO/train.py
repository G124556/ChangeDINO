import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch


from GDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

#
print("1111111111111111111111111111111")

model = load_model("GDINO/config/GroundingDINO-SwinT-OGC.py", "GDINO/weights/groundingdino_swint_ogc.pth")
# model = load_model("GDINO/config/GroundingDINO-SwinT-OGC.py")
IMAGE_PATH = "/data2/gaoyupeng/LESPS-master/ALL_data/whu-cd/train/pre/1006.png"
TEXT_PROMPT1="house"
TEXT_PROMPT = ["house","house","house","house","house"]
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)


image=torch.randn(5,3,512,512)

boxes, logits, phrases ,poss = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT1,
    targets=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image1.jpg", annotated_frame)