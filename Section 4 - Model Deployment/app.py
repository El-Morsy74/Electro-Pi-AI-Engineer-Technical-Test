import json
import torch
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

# ----------------------------------------------------
# 1. Configuration & 4-bit Quantized Model Setup
# ----------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if DEVICE == "cuda":
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quant_config, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="auto")

app = FastAPI(title="Qwen API")

# Request Schema
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    stream: bool = False

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

# ----------------------------------------------------
# 2. Generation Endpoint
# ----------------------------------------------------
@app.post("/generate")
def generate(req: PromptRequest):
    # System grounding for technical domain accuracy
    messages = [
        {"role": "system", "content": "You are a precise technical AI assistant. Answer factual computer science and machine learning questions accurately."},
        {"role": "user", "content": req.prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(DEVICE)

    # 1. Streaming Mode
    if req.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        Thread(target=model.generate, kwargs=dict(**inputs, streamer=streamer, max_new_tokens=req.max_tokens)).start()
        
        def stream():
            for token in streamer:
                yield f"data: {json.dumps({'content': token})}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(stream(), media_type="text/event-stream")

    # 2. Non-Streaming Mode
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_tokens)
    
    response_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)