import os
import gc
import sys
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from utils.toolbox import remove_image_extensions, block_timer

# Data configuration
# data_root = "datasets/Typographic-Dataset"
data_root = "datasets/typographic-word-invisible"
log_folder = "logs/typographic_word/invisible"

prefix_num = 0
prefix_strs = {
    0: "Answer with the option's letter from the given choices directly.", 
    1: "Focus on the visual aspects of the image, including colors, shapes, composition, and any notable visual themes. Answer with the option's letter from the given choices directly.",
    2: "Focus on the visual aspects of the image, including colors, shapes, composition, and any notable visual themes. Provide a detailed visual description of the image to answer the following question."
}

# image_folders = [os.path.join(data_root, folder) for folder in ['color-r0', 'counting-r0', 'species-r0', 'complex-r0']]
# image_folders = [os.path.join(data_root, folder) for folder in ['color-r1', 'counting-r1', 'species-r1', 'complex-r1']]
# image_folders = [os.path.join(data_root, folder) for folder in os.listdir(data_root) if any(task in folder for task in ['color-', 'counting-', 'species-', 'complex-'])]
image_folders = [os.path.join(data_root, folder) for folder in os.listdir(data_root) if any(task in folder for task in ['lowopacity', 'nearblack', 'nearwhite'])]

# Model configuration
model_root = "models"
model_paths = [
    os.path.join(model_root, "Ovis2.5-2B"),
    os.path.join(model_root, "Ovis2.5-9B"),
]

print(image_folders)
os.makedirs(log_folder, exist_ok=True)

# Main processing loop
model = None

for model_path in model_paths:
    # Cleanup previous model if exists
    if model is not None:
        del model
        model = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Load Ovis model
    print(f"\nLoading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()

    # Process each image folder
    for image_folder in image_folders:
        log_file = f"{os.path.basename(image_folder)}-{os.path.basename(model_path)}-prefix{prefix_num}.log"
        log_path = os.path.join(log_folder, log_file)
        
        if os.path.exists(log_path):
            print(f"Log file already exists, skipping: {log_path}")
            continue
        
        with block_timer(f"{os.path.basename(image_folder)} on {os.path.basename(model_path)}"):
            out = []
            
            for image_file in tqdm(os.listdir(image_folder)):
                # Prepare image path
                image_path = os.path.join(image_folder, image_file)
                
                # Determine question based on folder name or image file
                label = remove_image_extensions(image_file).split('-')[-2]
                mislabel = remove_image_extensions(image_file).split('-')[-1]
                challenge = remove_image_extensions(image_file).split('-')[-3].replace("%3F", "?")
                
                if "color" in image_folder or "complex" in image_folder:
                    question = f'{challenge} (a) {label} (b) {mislabel}'
                if "counting" in image_folder:
                    question = f'How many {challenge} are in the image? (a) {label} (b) {mislabel}'
                if "species" in image_folder:
                    question = f'What entity is depicted in the image? (a) {label} (b) {mislabel}'

                # Add prefix and handle 2-phase inference if needed
                if prefix_num == 2:
                    current_question = f"{prefix_strs[prefix_num]} {question}".strip()
                    
                    # Phase 1: Get visual description
                    messages_p1 = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.open(image_path)},
                            {"type": "text", "text": current_question},
                        ],
                    }]
                    
                    # Preprocess inputs for Phase 1
                    input_ids_p1, pixel_values_p1, grid_thws_p1 = model.preprocess_inputs(
                        messages=messages_p1,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    input_ids_p1 = input_ids_p1.cuda()
                    pixel_values_p1 = pixel_values_p1.cuda() if pixel_values_p1 is not None else None
                    grid_thws_p1 = grid_thws_p1.cuda() if grid_thws_p1 is not None else None
                    
                    # Generate Phase 1 output (visual description)
                    outputs_p1 = model.generate(
                        inputs=input_ids_p1,
                        pixel_values=pixel_values_p1,
                        grid_thws=grid_thws_p1,
                        enable_thinking=False,
                        enable_thinking_budget=False,
                        max_new_tokens=1024,
                    )
                    
                    description = model.text_tokenizer.decode(outputs_p1[0], skip_special_tokens=True)
                    
                    # Phase 2: Use description for final answer
                    question = f"Based on the image description: {description}. Answer with the option's letter from the given choices directly. {question}"
                else:
                    question = f"{prefix_strs[prefix_num]} {question}".strip()

                # Prepare messages in Ovis format
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.open(image_path)},
                        {"type": "text", "text": question},
                    ],
                }]

                # Preprocess inputs
                input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                    messages=messages,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                input_ids = input_ids.cuda()
                pixel_values = pixel_values.cuda() if pixel_values is not None else None
                grid_thws = grid_thws.cuda() if grid_thws is not None else None

                # Generate output
                outputs = model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    enable_thinking=False,
                    enable_thinking_budget=False,
                    max_new_tokens=128,
                )

                # Decode response
                answer = model.text_tokenizer.decode(outputs[0], skip_special_tokens=False)

                # Store result
                out.append({
                    'image_file': image_file,
                    'question': question,
                    'answer': answer
                })
                print(image_file)
                print(question)
                print(answer)

            # Write results to log file
            print(f"Writing results to {log_path}")
            with open(log_path, 'w') as f:
                for item in out:
                    f.write(json.dumps(item))
                    f.write("\n")