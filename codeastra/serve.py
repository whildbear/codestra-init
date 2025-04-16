import os
from ray import serve
from typing import Dict
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI
import logging

logger = logging.getLogger("ray.serve")
app = FastAPI()

@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine_args = engine_args
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_chat = None

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            self.openai_serving_chat = OpenAIServingChat(
                engine=self.engine,
                model_config=model_config,
                served_model_names=[self.engine_args.model],
                response_role="assistant",
            )

        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)

        if request.stream:
            return StreamingResponse(generator, media_type="text/event-stream")

        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    from vllm.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    args_list = [f"--{k}={v}" for k, v in cli_args.items()]
    parsed_args = parser.parse_args(args_list)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    return VLLMDeployment.bind(engine_args)


model = build_app({
    "model": os.environ.get("MODEL_ID", "rootxhacker/CodeAstra-7B"),
    "tensor-parallel-size": os.environ.get("TENSOR_PARALLELISM", "1"),
    "pipeline-parallel-size": os.environ.get("PIPELINE_PARALLELISM", "1"),
    "dtype": os.environ.get("VLLM_DTYPE", "float16"),
    "max-model-len": os.environ.get("MAX_MODEL_LEN", "2048")
})

