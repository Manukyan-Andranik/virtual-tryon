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
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List
import torch.nn as nn 
import torch
import os
from transformers import AutoTokenizer
import numpy as np
from dress_mask import get_mask_location
from torchvision import transforms
import base_net
from pose_parsing.humanparsing.run_parsing import Parsing
from pose_parsing.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"torch devices {torch.cuda.device_count()}")
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_name(1))
#exit()
def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


def get_base_path():
    config_path = os.path.expanduser('~/.config/myapp/config')
    with open(config_path, 'r') as file:
        config = file.read()
    base_path = config.split('=')[1].strip().strip("'\"")
    return base_path

base_path = 'yisol/IDM-VTON' #"/base" #get_base_path()

example_path = os.path.join(os.path.dirname(__file__), 'example')

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

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
    #device=device
    )

#my_unet = nn.DataParallel(UNet_Encoder, device_ids = [0,1])

#exit()
parsing_model = Parsing(gpu_id=1)
openpose_model = OpenPose(gpu_id=1)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)#.to(device)
text_encoder_two.requires_grad_(False)#.to(device)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )
pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,#.to(device,torch.float16),
        text_encoder_2 = text_encoder_two,#.to(device,torch.float16),
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
#        unet_encoder = UNet_Encoder
    #    device_map = device
        )
#pipe.unet = pipe.unet.cuda()
pipe.unet_encoder = UNet_Encoder.cuda()
pipe = pipe.to(device)
print("unet is cuda: ", next(pipe.unet.parameters()).is_cuda)
print("unet-encoder is cuda: ", next(pipe.unet_encoder.parameters()).is_cuda)
print("vae is cuda: ", next(pipe.vae.parameters()).is_cuda)
print("text_encoder is cuda: ", next(pipe.text_encoder.parameters()).is_cuda)
print("text_encoder_2 is cuda: ", next(pipe.text_encoder_2.parameters()).is_cuda)
#print("feature_extractor is cuda: ", next(pipe.feature_extractor.parameters()).is_cuda)
#print("tokenizer is cuda: ", next(pipe.tokenizer.parameters()).is_cuda)
#print("tokenizer 2 is cuda: ", next(pipe.tokenizer_2.parameters()).is_cuda)
#print("scheduler is cuda: ", next(pipe.scheduler.parameters()).is_cuda)
#print("image_encoder is cuda: ", next(pipe.image_encoder.parameters()).is_cuda)
print("Success_________")
#exit()
def start_tryon(person_img,garm_img,prompt,dropdown,category, is_checked,is_checked_crop,denoise_steps,seed):
     
    is_checked = True
    is_checked_crop = False
    denoise_steps = 32
    seed = 0
    #openpose_model.preprocessor.body_estimation.model#.to(device)
    #pipe.unet_encoder.to(device)

    garment_des = f"{dropdown} dress. {prompt}"
    category = "dresses"
    garm_img= garm_img.convert("RGB").resize((768,1024))
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
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))


    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('dc', category, model_parse, keypoints)
        #mask, mask_gray = get_mask_location('hd', "tops", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(person_img['layers'][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    

    args = base_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    print("____________args_____________")
    print(args)
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))

    with torch.no_grad():
        # Extract the images
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                print(f"--------flag-0-{device}---------")
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
#                        device = torch.device(device) 
                    
                        )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    print(f"___________flag-1-{device}-----------")
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
                            device = torch.device(device)
                            )



                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0)#.to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0)#.to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    print(f"___________flag-2-{device}-----------")
                    print("prompd embeds is cuda: ", prompt_embeds.is_cuda)
                    print("negative prompd embeds is cuda: ", negative_prompt_embeds.is_cuda)
                    print("pooled_prompt_embeds is cuda: ", pooled_prompt_embeds.is_cuda)
                    print("negative_pooled_prompt_embeds is cuda: ", negative_pooled_prompt_embeds.is_cuda)
                    print("prompt_embeds_c is cuda: ", prompt_embeds_c.is_cuda)
                    print("pose img is cuda: ", pose_img.is_cuda)
                    print("262 success")
                    exit()

                    images = pipe(
                        prompt_embeds=prompt_embeds,#.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds,#.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds,#.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,#.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img,#.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c,#.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
#                        device=device
                        )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig
    else:
        return images[0]
        #return images[0], mask_gray

# garm_list = os.listdir(os.path.join(example_path,"cloth"))
# garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

# human_list = os.listdir(os.path.join(example_path,"human"))
# human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

# human_ex_list = []
# for ex_human in human_list_path:
#     ex_dict= {}
#     ex_dict['background'] = ex_human
#     ex_dict['layers'] = None
#     ex_dict['composite'] = None
#     human_ex_list.append(ex_dict)





from flask import Flask, request, jsonify
import base64
from io import BytesIO

from PIL import Image
def image_to_base64(img):
    """Converts a PIL Image to a base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

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
    print("Starting tryon--------------------")
    res_image = start_tryon(person_img = person,
                garm_img = cloth,
                prompt = "",
                dropdown = "",
                category = position,
                is_checked = True,
                is_checked_crop = True,
                denoise_steps = 32,
                seed = 42)

    res_image_base64 = image_to_base64(res_image)

    return jsonify({"res_image": res_image_base64, "status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)


    
