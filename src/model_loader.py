from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME, DEVICE

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()