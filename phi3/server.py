import os
import logging
from typing import Dict, Optional

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.logger import RequestLogger
from vllm.utils import FlexibleArgumentParser

# Налаштування логування
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ray.serve")

app = FastAPI()

@serve.deployment(
    name="VLLMDeployment",
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    num_replicas=1,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: str = "default",
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine_args = engine_args
        self.response_role = response_role
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format

        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.openai_serving_chat = None

    async def _init_openai_serving_chat(self):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            served_model_names = (
                self.engine_args.served_model_name
                if self.engine_args.served_model_name
                else [self.engine_args.model]
            )
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names,
                self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
            )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        await self._init_openai_serving_chat()

        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(generator.model_dump(), status_code=generator.code)

        if request.stream:
            return StreamingResponse(generator, media_type="text/event-stream")

        return JSONResponse(generator.model_dump())

def parse_vllm_args(cli_args: Dict[str, str]):
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    arg_parser = FlexibleArgumentParser("vLLM OpenAI-Compatible API")
    parser = make_arg_parser(arg_parser)

    arg_strings = [f"--{key}={value}" for key, value in cli_args.items()]
    logger.info(f"CLI args: {arg_strings}")

    return parser.parse_args(arg_strings)

def build_app(cli_args: Dict[str, str]) -> serve.Application:
    # Використовуємо MODEL_ID із env_vars, якщо не задано в cli_args
    model_id = cli_args.get("model") or os.environ.get("MODEL_ID")
    if not model_id:
        logger.error("MODEL_ID must be specified in cli_args or environment variables")
        raise ValueError("MODEL_ID must be specified in cli_args or environment variables")
    
    # Оновлюємо cli_args із model_id
    cli_args = cli_args.copy()
    cli_args["model"] = model_id
    logger.debug(f"Building app with cli_args: {cli_args}")
    
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)

    # Конфігурація для 16 ГБ VRAM
    engine_args.worker_use_ray = True
    engine_args.dtype = os.environ.get("VLLM_DTYPE", "float16")
    engine_args.enforce_eager = True
    engine_args.trust_remote_code = True
    engine_args.max_num_seqs = 3  # Оптимізовано для стабільності
    engine_args.max_model_len = int(os.environ.get("MAX_MODEL_LEN", 2048))  # Використовуємо env_vars
    engine_args.gpu_memory_utilization = 0.85  # Зменшено для безпеки

    deployment = VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
        cli_args.get("chat_template_content_format", "default"),
    )
    logger.debug(f"Deployment object: {deployment}")
    return deployment
