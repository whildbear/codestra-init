from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@serve.deployment(
    name="VLLMDeployment",
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    num_replicas=1,
)
class Phi4Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-mini-instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
            worker_use_ray = True
            max_num_seqs = 3
            enforce_eager = True
            gpu_memory_utilization = 0.85
        )

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

serve.run(Phi4Model.bind())

