import base64
from fastapi import FastAPI, File, UploadFile
from typing import List
from PIL import Image
import io
import json_tricks as json
import yaml
import cv2
import numpy as np
from utils import *

app = FastAPI()

#load config
configs = yaml.load(open('configs/config.yaml', "r"), Loader=yaml.Loader)

# build model from configs
detector, pose_estimator, visualizer = build_model(configs)

@app.post("/process_image")
async def process_image(images: List[UploadFile] = File(...)):
    processed_images = []
    pred_result = []

    for image in images:
        buffer = io.BytesIO()
        # Open the uploaded image with Pillow
        image = Image.open(io.BytesIO(await image.read()))
        # Convert the image to a NumPy array
        image = np.array(image)
        
        # Process image
        pred_instances_list, vis_img = process_img(configs, image, detector,pose_estimator, visualizer)
        # Convert the processed image back to a Pillow image    
        processed_pil_image = Image.fromarray(vis_img)
        # Save the processed image to a BytesIO object
        processed_pil_image.save(buffer, format="JPEG")

        # Add the processed image to the list
        processed_images.append(base64.b64encode(buffer.getvalue()))
        pred_result.append(pred_instances_list)
    
    pred_result = json.dumps(
            dict(
                meta_info=pose_estimator.dataset_meta,
                instance_info=pred_result),
            indent='\t')
    # return response
    response = {"processed_images": processed_images,'pred_result': pred_result}

    return response

