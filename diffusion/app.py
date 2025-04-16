from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import Response
import torch
from ray import serve
from ray.serve.handle import DeploymentHandle

app = FastAPI()

@serve.deployment(
    ray_actor_options={
        "resources": {"GPU_worker": 1}
    },
    num_replicas=1
)
class StableDiffusion:
    def __init__(self):
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16
        )
        self.pipe.enable_attention_slicing()
        self.pipe.to("cuda")

    def generate(self, prompt: str, img_size: int = 512):
        image = self.pipe(prompt, height=img_size, width=img_size).images[0]
        return image

@serve.deployment
@serve.ingress(app)
class API:
    def __init__(self, model: DeploymentHandle):
        self.model = model

    @app.get("/imagine", response_class=Response)
    async def imagine(self, prompt: str, img_size: int = 512):
        image = await self.model.generate.remote(prompt, img_size)
        buf = BytesIO()
        image.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

entrypoint = API.bind(StableDiffusion.bind())

