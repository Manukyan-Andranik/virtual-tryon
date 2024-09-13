import sys
sys.path.append('./')
from PIL import Image
from base.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from base.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from base.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
from torch.nn import DataParallel
import os
from transformers import AutoTokenizer
import numpy as np
from dress_mask import get_mask_location
from torchvision import transforms
import base_net
from pose_parsing.humanparsing.run_parsing import Parsing
from pose_parsing.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device_ids = [0, 1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

# Инициализация моделей
unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)

tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)

text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)

vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)
UNet_Encoder = UNet_Encoder.cuda()
UNet_Encoder = DataParallel(UNet_Encoder, device_ids=device_ids)
# Перенос моделей на несколько GPU
unet = DataParallel(unet, device_ids=device_ids).cuda()
text_encoder_one = DataParallel(text_encoder_one, device_ids=device_ids).cuda()
text_encoder_two = DataParallel(text_encoder_two, device_ids=device_ids).cuda()
image_encoder = DataParallel(image_encoder, device_ids=device_ids).cuda()
vae = DataParallel(vae, device_ids=device_ids).cuda()
UNet_Encoder = DataParallel(UNet_Encoder, device_ids=device_ids).cuda()

parsing_model = DataParallel(Parsing(0), device_ids=device_ids).cuda()
openpose_model = DataParallel(OpenPose(0).preprocessor.body_estimation.model, device_ids=device_ids).cuda()

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet.module,
    vae=vae.module,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one.module,
    text_encoder_2=text_encoder_two.module,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder.module,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder.module

tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def start_tryon(person_img, garm_img, dress, category, is_checked, is_checked_crop, denoise_steps, seed):
    is_checked = True
    is_checked_crop = True
    denoise_steps = 32
    seed = 0

    garment_des = dress + ' dress'
    category = "dresses"
    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = person_img.convert("RGB")    
    
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

    if is_checked:
        human_img_tensor = tensor_transfrom(human_img.resize((384, 512))).unsqueeze(0).cuda()
        keypoints = openpose_model(human_img_tensor)
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('dc', category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(person_img.convert("RGB").resize((768, 1024)))
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    args = base_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)    
    pose_img = pose_img[:, :, ::-1]    
    pose_img = Image.fromarray(pose_img).resize((768, 1024))
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img = tensor_transfrom(pose_img).unsqueeze(0).cuda()
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).cuda()
                    
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    
                    images = pipe(
                        prompt_embeds=prompt_embeds.cuda(),
                        negative_prompt_embeds=negative_prompt_embeds.cuda(),
                        pooled_prompt_embeds=pooled_prompt_embeds.cuda(),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.cuda(),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.cuda(),
                        text_embeds_cloth=prompt_embeds_c.cuda(),
                        cloth=garm_tensor.cuda(),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768, 1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig
    else:
        return images[0]


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/virtual-dressing', methods=['POST'])
def virtual_dressing():
    person_bt = request.files.get('person')
    cloth_bt = request.files.get("cloth")
    person = Image.open(person_bt)
    cloth = Image.open(cloth_bt)
    position = request.form.get("position")
    full_body = request.form.get("full_body")
    desired_height = request.form.get("desired_height")
    print({
        "position": position,
        "full_body": full_body,
        "desired_height": desired_height,
        "person": person,
        "cloth": cloth
    })
    res_image = start_tryon(
        person_img=person, garm_img=cloth, dress="tshirt", category=position, 
        is_checked=True, is_checked_crop=False, denoise_steps=32, seed=0
    )
    print("res_image", type(res_image))
    return jsonify({"res_image": res_image, "status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

