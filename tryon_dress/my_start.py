import sys
sys.path.append('./')
import torch
import os
import numpy as np
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)
from diffusers import DDPMScheduler, AutoencoderKL
from torchvision import transforms
from base.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from base.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from base.unet_hacked_tryon import UNet2DConditionModel
from pose_parsing.humanparsing.run_parsing import Parsing
from pose_parsing.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from dress_mask import get_mask_location
from flask import Flask, request, jsonify
from typing import List

# Set up the device and multiple GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]  # Specify the GPUs you want to use (e.g., cuda:0 and cuda:1)

# Paths
base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

# Load and parallelize the models across GPUs
unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
), device_ids=device_ids).to(device)

vae = torch.nn.DataParallel(AutoencoderKL.from_pretrained(
    base_path,
    subfolder="vae",
    torch_dtype=torch.float16,
), device_ids=device_ids).to(device)

text_encoder_one = torch.nn.DataParallel(CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
), device_ids=device_ids).to(device)

text_encoder_two = torch.nn.DataParallel(CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
), device_ids=device_ids).to(device)

image_encoder = torch.nn.DataParallel(CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
), device_ids=device_ids).to(device)

UNet_Encoder = torch.nn.DataParallel(UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
), device_ids=device_ids).to(device)

# Load other components
tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
parsing_model = Parsing(0)
openpose_model = OpenPose(0)

# Set up the pipeline
pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)

pipe.unet_encoder = UNet_Encoder
pipe.to(device)

# Transformation function
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 1
    return Image.fromarray((mask * 255).astype(np.uint8))

# Main function for virtual try-on
def start_tryon(dict, garm_img, dress, category, is_checked, is_checked_crop, denoise_steps, seed):
    is_checked = True
    is_checked_crop = True
    denoise_steps = 32
    seed = 0
    openpose_model.preprocessor.body_estimation.model.to(device)
    
    garment_des = dress + ' dress'
    category = "dresses"
    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict.convert("RGB")

    # Handle cropping
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    # Process keypoints and mask if checked
    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('dc', category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # Perform inference on both GPUs
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = f"a photo of {garment_des}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            # Prepare pose image and garment tensor
            pose_img = tensor_transfrom(human_img).unsqueeze(0).to(device)
            garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device)
            generator = torch.Generator(device).manual_seed(seed)

            # Generate the final image
            images = pipe(
                prompt_embeds=prompt_embeds.to(device),
                negative_prompt_embeds=negative_prompt_embeds.to(device),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img.to(device),
                text_embeds_cloth=prompt_embeds.to(device),
                cloth=garm_tensor.to(device),
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                guidance_scale=2.0,
            )[0]

    return images[0]

# Flask app for virtual dressing API
app = Flask(__name__)

@app.route('/virtual-dressing', methods=['POST'])
def virtual_dressing():
    try:
        person = request.files.get('person')
        cloth = request.files.get("cloth")
        position = request.form.get("position")
        res_image = start_tryon(
            dict=person,
            garm_img=cloth,
            dress="dress",
            category=position,
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=32,
            seed=0
        )
        return jsonify({"res_image": res_image, "status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

