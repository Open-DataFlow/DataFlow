import pytest
from dataflow.serving import APIOrcaRouterServing


@pytest.mark.api
def test_orca_router_serving_defaults_and_request(dummy_server_base_url, monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "dummy-key")

    api_url = (
        f"{dummy_server_base_url}/v1/chat/completions"
        f"?queue=0&ka_interval=0.05&stream=0"
        f"&body=hello&think="
    )

    cli = APIOrcaRouterServing(
        api_url=api_url,
        model_name="orcarouter/auto",
        connect_timeout=1.0,
        read_timeout=3.0,
        max_retries=1,
        max_workers=1,
    )

    assert cli.api_url == api_url
    assert cli.model_name == "orcarouter/auto"
    assert cli.api_key == "dummy-key"

    _id, resp = cli._api_chat_with_id(
        id=0,
        payload=[{"role": "user", "content": "hi"}],
        model="orcarouter/auto",
        is_embedding=False,
    )

    assert _id == 0
    assert resp == "hello"

    cli.cleanup()


@pytest.mark.api
def test_orca_router_serving_requires_key(monkeypatch):
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        APIOrcaRouterServing()
