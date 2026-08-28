from .api_llm_serving_request import APILLMServing_request


class APIOrcaRouterServing(APILLMServing_request):
    """
    OpenAI-compatible serving class backed by the OrcaRouter gateway.

    OrcaRouter (https://www.orcarouter.ai) is an OpenAI-compatible AI gateway that,
    like OpenRouter, exposes a provider/model namespace across many models through a
    single endpoint. On top of that it adds adaptive routing, automatic failover,
    zero-markup inference, observability, guardrails, and agent-tool governance on
    the same OpenAI-compatible API.

    This class reuses the request/retry/formatting logic of APILLMServing_request and
    only wires it to OrcaRouter defaults, so DataFlow users can adopt the gateway
    without treating it as an anonymous custom base URL.
    """
    def __init__(self,
                 api_url: str = "https://api.orcarouter.ai/v1/chat/completions",
                 key_name_of_api_key: str = "ORCAROUTER_API_KEY",
                 model_name: str = "orcarouter/auto",
                 temperature: float = 0.0,
                 max_workers: int = 10,
                 max_retries: int = 5,
                 connect_timeout: float = 10.0,
                 read_timeout: float = 120.0,
                 **configs: dict):
        """
        Initialize OrcaRouter serving instance.

        Args:
            api_url: OrcaRouter OpenAI-compatible chat completions endpoint
            key_name_of_api_key: Environment variable holding the OrcaRouter API key
            model_name: OrcaRouter model namespace id (e.g. "orcarouter/auto")
            temperature: Sampling temperature
            max_workers: Number of concurrent workers for batch processing
            max_retries: Number of LLM inference retry chances for each input
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            **configs: Additional parameters forwarded to the API payload

        Note:
            Set the API key via `export ORCAROUTER_API_KEY=sk-orca-...` before use.
        """
        super().__init__(
            api_url=api_url,
            key_name_of_api_key=key_name_of_api_key,
            model_name=model_name,
            temperature=temperature,
            max_workers=max_workers,
            max_retries=max_retries,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            **configs,
        )
