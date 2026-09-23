import os
import gc
import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

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
    os.path.join(model_root, "gemma-3-4b-it"),
    os.path.join(model_root, "gemma-3-12b-it"),
    os.path.join(model_root, "gemma-3-27b-it"),
]

print(image_folders)
os.makedirs(log_folder, exist_ok=True)

# Main processing loop
model = None
processor = None

for model_path in model_paths:
    # Cleanup previous model if exists
    if model is not None:
        del model
        del processor
        model = None
        processor = None
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load Gemma3 model and processor
    print(f"\nLoading model from {model_path}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)

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
                    messages_p1 = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": current_question}
                            ]
                        }
                    ]
                    
                    inputs_p1 = processor.apply_chat_template(
                        messages_p1,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(model.device, dtype=torch.bfloat16)

                    input_len_p1 = inputs_p1["input_ids"].shape[-1]

                    with torch.inference_mode():
                        generation_p1 = model.generate(
                            **inputs_p1, 
                            max_new_tokens=1024,
                            do_sample=False
                        )
                        generation_p1 = generation_p1[0][input_len_p1:]

                    description = processor.decode(generation_p1, skip_special_tokens=True)
                    
                    # Phase 2: Use description for final answer
                    question = f"Based on the image description: {description}. Answer with the option's letter from the given choices directly. {question}"
                else:
                    question = f"{prefix_strs[prefix_num]} {question}".strip()

                # Prepare messages in Gemma3 format
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": question}
                        ]
                    }
                ]

                # Prepare inputs
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)

                input_len = inputs["input_ids"].shape[-1]

                # Generate output
                with torch.inference_mode():
                    generation = model.generate(
                        **inputs, 
                        max_new_tokens=128,
                        do_sample=False
                    )
                    generation = generation[0][input_len:]

                # Decode the generated tokens
                answer = processor.decode(generation, skip_special_tokens=True)

                # Store result
                out.append({
                    'image_file': image_file,
                    'question': question,
                    'answer': answer
                })
                print(f"{image_file}")
                print(f"{question}")
                print(f"{answer}\n")

            # Write results to log file
            print(f"Writing results to {log_path}")
            with open(log_path, 'w') as f:
                for item in out:
                    f.write(json.dumps(item))
                    f.write("\n")