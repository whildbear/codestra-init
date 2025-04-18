from ray import serve
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@serve.deployment(
    name="VLLMDeployment",
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    num_replicas=1,
)
@serve.ingress(app)
class Phi4Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-mini-instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    @app.post("/generate")
    async def generate(self, prompt: Prompt):
        inputs = self.tokenizer(prompt.prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": response}

