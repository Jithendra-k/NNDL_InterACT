from ultralytics import YOLO
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

#Hides warnings.
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#=============================================================================
#Loading Finetuned Llama chat model

# Load the fine-tuned model and tokenizer
model_name = "Jithendra-k/interACT_LLM"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the input prompt
prompt = ("I want to send an email")

# Run text generation pipeline with the model
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=27, do_sample=True)
result = pipe(f"<s>[INST] {prompt} [/INST]")

# Print the generated text
print(result[0]['generated_text'])

res_string = result[0]['generated_text']

# Split the string based on the [INST] and [/INST] tags
parts = res_string.split("[INST]")[1].split("[/INST]")

class_list = parts[1].strip()
words = class_list.split(",")

words = [word.strip().lower() for word in words if word.strip()]
classes = set(words)
classes = list(classes)
print(classes)
#=============================================================================
#Initialize a YOLO-World model
model = YOLO('./Build_Yolo-Agent/yolov8s-worldv2.pt').to(device)  # loading model weights

# Define custom classes
model.set_classes(classes)

# Execute prediction for specified categories on an image
results = model.predict('./Build_Yolo-Agent/test-room.jpeg')

# Show results
results[0].show()
#===============================================================================
