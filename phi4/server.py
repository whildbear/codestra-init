from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

@serve.deployment(
    name="Phi4Model",
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    num_replicas=1,
)
class Phi4Model:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-mini-instruct",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256

@serve.deployment
@serve.ingress(app)
class API:
    def __init__(self, model: DeploymentHandle):
        self.model = model

    @app.post("/generate")
    async def generate_text(self, request: PromptRequest):
        response = await self.model.generate.remote(
            request.prompt,
            request.max_new_tokens
        )
        return {"response": response}

entrypoint = API.bind(Phi4Model.bind())

