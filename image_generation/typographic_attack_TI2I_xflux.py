import os
import sys
sys.path.append("x-flux")

import shutil
from tqdm import tqdm
from PIL import Image

from src.flux.xflux_pipeline import XFluxPipeline


def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text

def combine_images(original, generated):
    width = original.width + generated.width
    height = max(original.height, generated.height)
    combined = Image.new('RGB', (width, height))
    combined.paste(original, (0, 0))
    combined.paste(generated, (original.width, 0))
    return combined

def main():
    seed = 42
    image_size = 512
    images_num = 1
    model_name = "xflux"
    device = "cuda"
    
    typos = ["naked", "bloody", "african", "asian", "glasses", "hat", "clean"]
    
    conditions = ["fringe"]
    exceptions = ["imperceptible"]

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
    xflux_pipeline = XFluxPipeline(model_type="flux-dev", device=device)
    print('loading ip-adapter...')
    xflux_pipeline.set_ip(repo_id="XLabs-AI/flux-ip-adapter", name="ip_adapter.safetensors")

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
                image_prompt = Image.open(os.path.join(dataset_root, source_dir, img_file))

                for i in range(images_num):
                    result = xflux_pipeline(
                        prompt=prompt,
                        neg_prompt=negative_prompt,
                        width=image_size,
                        height=image_size,
                        guidance=4,
                        num_steps=25,
                        seed=seed,
                        true_gs=3.5,
                        control_weight=0.8,
                        timestep_to_start_cfg=5,
                        image_prompt=image_prompt,
                        ip_scale=1.0,
                        neg_ip_scale=1.0,
                    )
                    seed = seed + 1

                    combined_image = combine_images(image_prompt, result)
                    combined_image.save(os.path.join(output_root, model_name, f'{source_dir}-{prompt[:50]}', remove_image_extensions(img_file)+f"-{i}.png"))


if __name__ == "__main__":
    main()