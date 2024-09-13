import os
import sys
sys.path.append('./')
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)
from diffusers import DDPMScheduler, AutoencoderKL
from base.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from base.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from base.unet_hacked_tryon import UNet2DConditionModel
from pose_parsing.humanparsing.run_parsing import Parsing
from pose_parsing.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from typing import List
from dress_mask import get_mask_location
import base_net
from torchvision.transforms.functional import to_pil_image

# Set device IDs for multiple GPUs
device_ids = [0, 1] if torch.cuda.is_available() else []

# Set device
device =  torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# Function to clear GPU memory
torch.cuda.empty_cache()

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 1
    mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask)

# Load Models and Tokenizers
base_path = 'yisol/IDM-VTON'

unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

# Freeze Parameters (ensure models do not consume memory for gradient storage)
UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

# Load pipeline and configure to half precision
tensor_transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

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

# DataParallel for multiple GPUs
if len(device_ids) > 1:
    pipe = torch.nn.DataParallel(pipe, device_ids=device_ids)
    openpose_model.preprocessor.body_estimation.model = torch.nn.DataParallel(openpose_model.preprocessor.body_estimation.model, device_ids=device_ids)

# Send models to the appropriate device
pipe_module = pipe.module if isinstance(pipe, torch.nn.DataParallel) else pipe
pipe_module.to(device)
pipe_module.unet_encoder.to(device)

# Set the environment variable to manage memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("Start function")
def start_tryon(person_img, garm_img, dress, category, is_checked, is_checked_crop, denoise_steps, seed):
    # Clear any cached CUDA memory
    #torch.cuda.empty_cache()

    # Optimize model execution with torch.no_grad()
    with torch.no_grad():
        garm_img = garm_img.convert("RGB").resize((768, 1024))
        human_img_orig = person_img.convert("RGB")
        # Crop human image if necessary
        print(type(garm_img))
        print(type(human_img_orig))
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
        print(human_img_orig) 
        # Parse and get mask
        if is_checked:
            keypoints = openpose_model(human_img.resize((384, 512)))
            model_parse, _ = parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = get_mask_location('dc', category, model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            mask = pil_to_binary_mask(person_img.convert("RGB").resize((768, 1024)))
        print("__________________________1")
        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        # Apply OpenPose for pose estimation
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        
        args = base_net.create_argument_parser().parse_args((
            'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'
        ))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))
        print("__________________________2")
        # Prompt Embeddings for Try-On Model
        prompt = f"model is wearing {dress} dress"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        with torch.cuda.amp.autocast(True):  # Mixed precision inference
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe_module.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt
            )

            pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

            images = pipe_module(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img.to(device, torch.float16),
                cloth=garm_tensor.to(device, torch.float16),
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=2.0
            )[0]

    if is_checked_crop:
        images = images.resize(crop_size)

    return images
print("app")
from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)

@app.route('/virtual-dressing', methods=['POST'])
def virtual_dressing():
    
    person_file = request.files.get('person')        
    cloth_file = request.files.get("cloth")
    position = request.form.get("position")
    full_body = request.form.get("full_body")
    desired_height = request.form.get("desired_height")
    
    person_bt = person_file.read() if person_file else None
    cloth_bt = cloth_file.read() if cloth_file else None

    if not person_bt or not cloth_bt:
        return jsonify({"error": "Person or cloth image is missing"}), 400

    # Open images from bytes
    person_image = Image.open(BytesIO(person_bt))
    cloth_image = Image.open(BytesIO(cloth_bt))
    
    print({
            "position": position,
            "full_body": full_body,
            "desired_height": desired_height,

            "person": person_image,
            "cloth": cloth_image
        })

    print("-"*20 + "inputs: success")
    
    res_image = start_tryon(
            person_img=person_image, garm_img=cloth_image, dress="tshirt", category=position, 
            is_checked=False, is_checked_crop=False, denoise_steps=32, seed=0
        )
    print("res_image", type(res_image))
     
         # Encode image to base64
    buffered = BytesIO()
    res_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({"res_image": img_str, "status": "success"}), 200
        #return jsonify({"res_image": "success", "status": "success"}), 200
    #except Exception as e:
     #   return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
