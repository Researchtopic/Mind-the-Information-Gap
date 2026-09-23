import os
import gc
import sys
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from utils.toolbox import remove_image_extensions, block_timer

SYSTEM_PROMPT = 'You are Ministral-3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYou power an AI assistant called Le Chat.\nYour knowledge base was last updated on 2023-10-01.\n\nWhen you\'re not sure about some information or when the user\'s request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don\'t have the information and avoid making up anything.\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").\nYou are always very attentive to dates, in particular you try to resolve dates and when asked about information at specific dates, you discard information that is at another date.\nYou follow these instructions in all languages, and always respond to the user in the language they use or request.\nNext sections describe the capabilities that you have.\n\n# WEB BROWSING INSTRUCTIONS\n\nYou cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\n\n# MULTI-MODAL INSTRUCTIONS\n\nYou have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.\nYou cannot read nor transcribe audio files or videos.\n\n# TOOL CALLING INSTRUCTIONS\n\nYou may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:\n\n1. When the request requires up-to-date information.\n2. When the request requires specific data that you do not have in your knowledge base.\n3. When the request involves actions that you cannot perform without tools.\n\nAlways prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment.'

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
    os.path.join(model_root, "Ministral-3-3B-Instruct-2512-BF16"),
    os.path.join(model_root, "Ministral-3-8B-Instruct-2512-BF16"),
    os.path.join(model_root, "Ministral-3-14B-Instruct-2512-BF16"),
]

print(image_folders)
os.makedirs(log_folder, exist_ok=True)

# Helper function for tokenization and moving to device
def prepare_tokenized_inputs(tokenizer, messages):
    tokenized = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        return_dict=True,
    )
    for key in tokenized:
        if isinstance(tokenized[key], torch.Tensor):
            if key == "pixel_values":
                tokenized[key] = tokenized[key].to(dtype=torch.bfloat16, device="cuda")
            else:
                tokenized[key] = tokenized[key].to(device="cuda")
    return tokenized

# Main processing loop
model = None
tokenizer = None

for model_path in model_paths:
    # Cleanup previous model if exists
    if model is not None:
        del model
        del tokenizer
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Load Mistral3 model and tokenizer
    print(f"\nLoading model from {model_path}")
    tokenizer = MistralCommonBackend.from_pretrained(model_path)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

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

                # Convert local image path to file URL format
                image_url = f"file://{image_path}"

                # Add prefix and handle 2-phase inference if needed
                if prefix_num == 2:
                    current_question = f"{prefix_strs[prefix_num]} {question}".strip()
                    
                    # Phase 1: Get visual description
                    messages_p1 = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": SYSTEM_PROMPT},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": current_question},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        },
                    ]
                    
                    tokenized_p1 = prepare_tokenized_inputs(tokenizer, messages_p1)
                    image_sizes_p1 = [tokenized_p1["pixel_values"].shape[-2:]]
                    
                    with torch.inference_mode():
                        output_p1 = model.generate(
                            **tokenized_p1,
                            image_sizes=image_sizes_p1,
                            max_new_tokens=1024,
                        )[0]
                    
                    description = tokenizer.decode(output_p1[len(tokenized_p1["input_ids"][0]):], skip_special_tokens=True)
                    
                    # Phase 2: Use description for final answer
                    question = f"Based on the image description: {description}. Answer with the option's letter from the given choices directly. {question}"
                else:
                    question = f"{prefix_strs[prefix_num]} {question}".strip()

                # Prepare messages in Mistral format
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ]

                # Tokenize inputs
                tokenized = prepare_tokenized_inputs(tokenizer, messages)
                image_sizes = [tokenized["pixel_values"].shape[-2:]]

                # Generate output
                with torch.inference_mode():
                    output = model.generate(
                        **tokenized,
                        image_sizes=image_sizes,
                        max_new_tokens=128,
                    )[0]

                # Decode only the generated part (trim input tokens)
                answer = tokenizer.decode(output[len(tokenized["input_ids"][0]):])

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