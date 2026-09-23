import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import time
import shutil
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers import AutoPipelineForImage2Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import AutoPipelineForText2Image


def combine_images(original, generated):
    width = original.width + generated.width
    height = max(original.height, generated.height)
    combined = Image.new('RGB', (width, height))
    combined.paste(original, (0, 0))
    combined.paste(generated, (original.width, 0))
    return combined

def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text

def main(args):
    image_size = (512, 512)
    images_num = 1
    model_name = args.model

    typos = ["naked", "bloody", "african", "asian", "glasses", "hat", "clean"]
    
    conditions = []
    exceptions = ["imperceptible", "fringe"]

    dataset_root = "datasets"
    output_root = "outputs"

    prompts = ["analog film photo, faded film, desaturated, 35mm photo",]
    # prompts = ["a youthful figure on the stage, full body view, dynamic pose",]

    negative_prompt = "blurry, monochrome, low resolution, worst quality, low quality, out of frame, watermark, signature, text, oversaturated, overexposed, underexposed, grainy, noisy, duplicate, 3d"

    source_dirs = []
    for item in os.listdir(dataset_root):
        for typo in typos:
            if typo in item and all(condition in item for condition in conditions) and not any(exception in item for exception in exceptions):
                source_dirs.append(item)
    print(source_dirs)

    # init model
    if model_name == 'sd1.5':
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("models/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None).to("cuda")
    if model_name == 'sdxl':
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("models/stable-diffusion-xl-refiner-1.0", variant="fp16", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    if model_name == 'sd3':
        from diffusers import StableDiffusion3Img2ImgPipeline
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("models/stable-diffusion-3-medium-diffusers", variant="fp16", torch_dtype=torch.float16, use_safetensors=True, device_map="balanced")
    if model_name == 'unclip':
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("models/stable-diffusion-2-1-unclip", variation="fp16", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    if model_name == 'kandinsky-2-1':
        pipe = AutoPipelineForImage2Image.from_pretrained("models/kandinsky-2-1", torch_dtype=torch.float16).to("cuda")
    if model_name == 'kandinsky-2-2':
        pipe = AutoPipelineForImage2Image.from_pretrained("models/kandinsky-2-2-decoder", torch_dtype=torch.float16).to("cuda")
    if model_name == 'kandinsky-3':
        pipe = AutoPipelineForImage2Image.from_pretrained("models/kandinsky-3", variant="fp16", torch_dtype=torch.float16, device_map="balanced")
    if model_name == 'ipadapter-sdxl':
        pipe = AutoPipelineForText2Image.from_pretrained("models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
        pipe.load_ip_adapter("models/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.set_ip_adapter_scale(0.5)
    if model_name == 'ipadapter-sd1.5':
        pipe = AutoPipelineForText2Image.from_pretrained("models/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to("cuda")
        pipe.load_ip_adapter("models/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipe.set_ip_adapter_scale(0.5)
    if model_name == 'ipadapter-sd3.5':
        # conda activate xflux
        import sys
        sys.path.append('models/SD3.5-Large-IP-Adapter')
        from models.transformer_sd3 import SD3Transformer2DModel
        from pipeline_stable_diffusion_3_ipa import StableDiffusion3Pipeline
        transformer = SD3Transformer2DModel.from_pretrained('models/stable-diffusion-3.5-large', subfolder="transformer", torch_dtype=torch.bfloat16)
        pipe = StableDiffusion3Pipeline.from_pretrained('models/stable-diffusion-3.5-large', transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
        pipe.init_ipadapter(
            ip_adapter_path='models/SD3.5-Large-IP-Adapter/ip-adapter.bin', 
            image_encoder_path='models/siglip-so400m-patch14-384', 
            nb_token=64, 
        )
    if model_name == 'ipadapter-instruct-sd3':
        # conda activate ipadapter-instruct
        import sys
        sys.path.append('IP-Adapter-Instruct')
        from ip_adapter.ip_adapter_instruct import IPAdapter_sd3_Instruct
        from ip_adapter.pipeline_stable_diffusion_sd3_extra_cfg import StableDiffusion3PipelineExtraCFG
        pipe = StableDiffusion3PipelineExtraCFG.from_pretrained("models/stable-diffusion-3-medium-diffusers", feature_extractor=None, safety_checker=None, torch_dtype=torch.float16).to("cuda")
        # pipe.enable_model_cpu_offload()
        ip_model = IPAdapter_sd3_Instruct(pipe, "models/CLIP-ViT-H-14-laion2B-s32B-b79K", "models/IP-Adapter-Instruct/ip-adapter-instruct-sd3.bin", "cuda", num_tokens=16)
    if model_name == 'kolors':
        # conda activate kolors
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_ipadapter import StableDiffusionXLPipeline
        from kolors.models.modeling_chatglm import ChatGLMModel
        from kolors.models.tokenization_chatglm import ChatGLMTokenizer
        from diffusers import  AutoencoderKL
        from kolors.models.unet_2d_condition import UNet2DConditionModel
        from diffusers import EulerDiscreteScheduler

        ckpt_dir = f'Kolors/weights/Kolors'
        text_encoder = ChatGLMModel.from_pretrained(f'{ckpt_dir}/text_encoder', torch_dtype=torch.float16).half()
        tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
        vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
        scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
        unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained( f'Kolors/weights/Kolors-IP-Adapter-Plus/image_encoder',  ignore_mismatched_sizes=True).to(dtype=torch.float16)
        ip_img_size = 336
        clip_image_processor = CLIPImageProcessor(size=ip_img_size, crop_size=ip_img_size)

        pipe = StableDiffusionXLPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, image_encoder=image_encoder, feature_extractor=clip_image_processor, force_zeros_for_empty_prompt=False)
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        
        if hasattr(pipe.unet, 'encoder_hid_proj'):
            pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj
        
        pipe.load_ip_adapter( f'Kolors/weights/Kolors-IP-Adapter-Plus' , subfolder="", weight_name=["ip_adapter_plus_general.bin"])
        pipe.set_ip_adapter_scale([0.5])
    if model_name == 'instantx-flux':
        # conda activate xflux
        import sys
        sys.path.append("FLUX.1-dev-IP-Adapter")

        from pipeline_flux_ipa import FluxPipeline
        from transformer_flux import FluxTransformer2DModel
        from infer_flux_ipa_siglip import IPAdapter

        image_encoder_path = "models/siglip-so400m-patch14-384"
        ipadapter_path = "models/FLUX.1-dev-IP-Adapter/ip-adapter.bin"
        transformer = FluxTransformer2DModel.from_pretrained("models/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16)

        pipe = FluxPipeline.from_pretrained("models/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16)
        ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)
    if model_name == 'instant-style':
        # conda activate xflux
        from diffusers import StableDiffusionXLPipeline

        # load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "models/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to("cuda")

        # load ip-adapter
        pipe.load_ip_adapter("models/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.enable_vae_tiling()

        # for style blocks only
        scale = {
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipe.set_ip_adapter_scale(scale)
    if model_name == 'ipadapter-composition':
        pipe = AutoPipelineForText2Image.from_pretrained("models/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to("cuda")
        pipe.load_ip_adapter("models/IP-Adapter", subfolder="models", weight_name="ip_plus_composition_sd15.safetensors")
        pipe.set_ip_adapter_scale(0.5)


    pipe.set_progress_bar_config(leave=False)
    pipe.set_progress_bar_config(disable=True)

    # generate images
    for source_dir in source_dirs:
        for prompt in prompts:
            print(f"current dir: {source_dir}.    current prompt: {prompt}.")

            # init output folders
            output_dir = os.path.join(output_root, model_name, f'{source_dir}-{prompt[:50]}')
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        
            img_files = sorted(os.listdir(os.path.join(dataset_root, source_dir)))
            for img_file in tqdm(img_files):
                init_image = Image.open(os.path.join(dataset_root, source_dir, img_file)).convert("RGB")
                init_image = init_image.resize(image_size, Image.Resampling.LANCZOS)
                
                for i in range(images_num):
                    if model_name == 'sd1.5':
                        images = pipe(prompt, negative_prompt=negative_prompt, image=init_image, strength=0.7, guidance_scale=7.5, num_inference_steps=20).images[0]
                    if model_name == 'sdxl':
                        images = pipe(prompt, negative_prompt=negative_prompt, image=init_image, strength=0.6, guidance_scale=5.0, num_inference_steps=20).images[0]
                    if model_name == 'sd3':
                        images = pipe(prompt, negative_prompt=negative_prompt, image=init_image, strength=0.6, guidance_scale=5.0, num_inference_steps=20).images[0]
                    if model_name == 'unclip':
                        init_image = init_image.resize((768, 768), Image.Resampling.LANCZOS)
                        images = pipe(init_image, prompt=prompt, negative_prompt=negative_prompt, guidance_scale=10, num_inference_steps=20).images[0]
                    if model_name == 'kandinsky-2-1':
                        images = pipe(prompt, negative_prompt=negative_prompt, image=init_image, strength=0.3, guidance_scale=4.0, num_inference_steps=20).images[0]
                    if model_name == 'kandinsky-2-2':
                        images = pipe(prompt, negative_prompt=negative_prompt, image=init_image, strength=0.3, guidance_scale=4.0, num_inference_steps=20).images[0]
                    if model_name == 'kandinsky-3':
                        images = pipe(prompt, negative_prompt=negative_prompt, image=init_image, strength=0.6, guidance_scale=3.0, num_inference_steps=20).images[0]
                    if model_name == 'ipadapter-sdxl':
                        init_image = init_image.resize((1024, 1024), Image.Resampling.LANCZOS)
                        images = pipe(prompt=prompt, ip_adapter_image=init_image, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
                    if model_name == 'ipadapter-sd1.5':
                        init_image = init_image.resize((512, 512), Image.Resampling.LANCZOS)
                        images = pipe(prompt=prompt, ip_adapter_image=init_image, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
                    if model_name == 'ipadapter-sd3.5':
                        init_image = init_image.resize((1024, 1024), Image.Resampling.LANCZOS)
                        images = pipe(
                            width=1024,
                            height=1024,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=24, 
                            guidance_scale=5.0,
                            clip_image=init_image,
                            ipadapter_scale=0.5,
                        ).images[0]
                    if model_name == 'ipadapter-instruct-sd3':
                        init_image = init_image.resize((1024, 1024), Image.Resampling.LANCZOS)
                        images = ip_model.generate(prompt=prompt, negative_prompt=negative_prompt, pil_image=init_image, num_samples=1, num_inference_steps=30, scale=0.8, query="use the face")[0]
                    if model_name == 'kolors':
                        init_image = init_image.resize((1024, 1024), Image.Resampling.LANCZOS)
                        images = pipe(
                            prompt=prompt,
                            ip_adapter_image=[init_image],
                            negative_prompt=negative_prompt, 
                            height=1024,
                            width=1024,
                            num_inference_steps=50, 
                            guidance_scale=5.0,
                            num_images_per_prompt=1,
                        ).images[0]
                    if model_name == 'instantx-flux':
                        images = ip_model.generate(
                            pil_image=init_image, 
                            prompt=prompt,
                            scale=0.7,
                            width=512, 
                            height=512,
                        )[0]
                    if model_name == 'instant-style':
                        init_image = init_image.resize((1024, 1024), Image.Resampling.LANCZOS)
                        images = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            ip_adapter_image=init_image,
                            guidance_scale=6.5,
                            num_inference_steps=25,
                        ).images[0]
                    if model_name == 'ipadapter-composition':
                        init_image = init_image.resize((512, 512), Image.Resampling.LANCZOS)
                        images = pipe(prompt=prompt, ip_adapter_image=init_image, negative_prompt=negative_prompt, num_inference_steps=50).images[0]

                    combined_image = combine_images(init_image, images)            
                    combined_image.save(os.path.join(output_root, model_name, f'{source_dir}-{prompt[:50]}', remove_image_extensions(img_file)+f"-{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ipadapter-composition')
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"total time spent: {(end_time - start_time) / 3600}h")