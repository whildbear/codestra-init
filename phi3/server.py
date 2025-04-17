import os
import logging
from typing import Dict, Optional, List

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import LoRAModulePath, PromptAdapterPath
from vllm.entrypoints.logger import RequestLogger
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger("ray.serve")
app = FastAPI()

@serve.deployment(
    name="VLLMDeployment",
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 4,
    },
    num_replicas=1,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template

        try:
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Engine init failed: {e}")

        self.openai_serving_chat = None

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            served_model_names = (
                self.engine_args.served_model_name
                if self.engine_args.served_model_name is not None
                else [self.engine_args.model]
            )
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names,
                self.response_role,
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    @app.post("/chat")
    async def prompt_only(self, request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            return JSONResponse({"error": "Missing 'prompt'"}, status_code=400)

        sampling_params = {
            "max_tokens": 256,
            "temperature": 0.7,
        }

        logger.info(f"Custom prompt request: {prompt}")
        result = await self.engine.generate(prompt, sampling_params)
        return JSONResponse({"response": result[0].outputs[0].text})


def parse_vllm_args(cli_args: Dict[str, str]):
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)

    # 🔧 Кастомні параметри для запуску на GPU (T4, A10)
    engine_args.worker_use_ray = True
    engine_args.dtype = "float16"
    engine_args.enforce_eager = True
    engine_args.trust_remote_code = True
    engine_args.max_model_len = 4092
    engine_args.gpu_memory_utilization = 0.9

    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )


model = build_app({
    "model": os.environ["MODEL_ID"],
    "tensor-parallel-size": os.environ.get("TENSOR_PARALLELISM", "1"),
    "pipeline-parallel-size": os.environ.get("PIPELINE_PARALLELISM", "1")
})

