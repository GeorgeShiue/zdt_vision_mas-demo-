import json
import logging
from typing import Any, Optional, Union

import requests

from autogen.code_utils import content_str
from autogen.formatting_utils import colored
from autogen.import_utils import optional_import_block, require_optional_import
from autogen.llm_config import LLMConfig
from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.img_utils import get_image_data, llava_formatter
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

with optional_import_block():
    import replicate

logger = logging.getLogger(__name__)

# we will override the following variables later.
SEP = "###"

DEFAULT_LLAVA_SYS_MSG = "You are an AI agent and you can view images."


class OllamaLLaVAAgent(MultimodalConversableAgent):
    def __init__(
        self,
        name: str,
        system_message: Optional[tuple[str, list]] = DEFAULT_LLAVA_SYS_MSG,
        *args,
        **kwargs: Any,
    ):
        """Args:
        name (str): agent name.
        system_message (str): system message for the ChatCompletion inference.
            Please override this attribute if you want to reprogram the agent.
        **kwargs (dict): Please refer to other kwargs in
            [ConversableAgent](/docs/api-reference/autogen/ConversableAgent#conversableagent).
        """
        super().__init__(
            name,
            system_message=system_message,
            *args,
            **kwargs,
        )

        assert self.llm_config is not None, "llm_config must be provided."
        self.register_reply([Agent, None], reply_func=OllamaLLaVAAgent._image_reply, position=2)

    def _image_reply(self, messages=None, sender=None, config=None):
        # Note: we did not use "llm_config" yet.

        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        # The formats for LLaVA and GPT are different. So, we manually handle them here.
        images = []
        prompt = content_str(self.system_message) + "\n"
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            # pdb.set_trace()
            images += [d["image_url"]["url"] for d in msg["content"] if d["type"] == "image_url"]
            content_prompt = content_str(msg["content"])
            prompt += f"{SEP}{role}: {content_prompt}\n"
        prompt += "\n" + SEP + "Assistant: "

        # TODO: PIL to base64
        images = [get_image_data(im) for im in images]
        print(colored(prompt, "blue"))

        out = ""
        retry = 10
        while len(out) == 0 and retry > 0:
            # image names will be inferred automatically from llava_call
            out = llava_call_binary(
                prompt=prompt,
                images=images,
                config_list=self.llm_config["config_list"],
                temperature=self.llm_config.get("temperature", 0.5),
                max_new_tokens=self.llm_config.get("max_new_tokens", 2000),
            )
            retry -= 1

        assert out != "", "Empty response from LLaVA."

        return True, out

def parse_ndjson_chunk(chunk: bytes) -> str:
    output = ""
    try:
        lines = chunk.decode("utf-8").strip().split("\n")
        for line in lines:
            data = json.loads(line)
            output += data.get("response", "")
    except Exception as e:
        print("Failed to parse chunk:", e)
    return output

@require_optional_import("replicate", "lmm")
def _llava_call_binary_with_config(
    prompt: str,
    images: list[Any],
    config: dict[str, Any],
    max_new_tokens: int = 1000,
    temperature: float = 0.5,
    seed: int = 1,
):
    pload = {
        "model": config["model"],
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "stop": SEP,
        "images": images,
    }
    response = requests.post(
        str(config["base_url"]).rstrip("/") + "/api/generate", json=pload, stream=True
    )

    output = ""
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            output += parse_ndjson_chunk(chunk)
    output = output.replace(prompt, "").strip().rstrip()
    return output


@require_optional_import("replicate", "lmm")
def llava_call_binary(
    prompt: str,
    images: list[Any],
    config_list: list[dict[str, Any]],
    max_new_tokens: int = 1000,
    temperature: float = 0.5,
    seed: int = 1,
):
    # TODO 1: add caching around the LLaVA call to save compute and cost
    # TODO 2: add `seed` to ensure reproducibility. The seed is not working now.

    for config in config_list:
        try:
            return _llava_call_binary_with_config(prompt, images, config, max_new_tokens, temperature, seed)
        except Exception as e:
            print(f"Error: {e}")
            continue


def llava_call(prompt: str, llm_config: Union[LLMConfig, dict]) -> str:
    """Makes a call to the LLaVA service to generate text based on a given prompt"""
    prompt, images = llava_formatter(prompt, order_image_tokens=False)

    for im in images:
        if len(im) == 0:
            raise RuntimeError("An image is empty!")

    return llava_call_binary(
        prompt,
        images,
        config_list=llm_config["config_list"],
        max_new_tokens=llm_config.get("max_new_tokens", 2000),
        temperature=llm_config.get("temperature", 0.5),
        seed=llm_config.get("seed"),
    )
