# Use a local Codex sign-in as PentAGI's LLM provider

`examples/codex-bridge/server.py` is an experimental local adapter. It starts `codex app-server` using the ChatGPT sign-in already held by the Codex CLI on the same computer, then exposes the Chat Completions endpoints required by PentAGI's Custom provider. It does not read or copy Codex's credential files, and it does not need an OpenAI API key. A Codex desktop login alone may not be shared with the CLI on every installation; run `codex login status` first and, if necessary, sign in to the CLI once.

The adapter converts each PentAGI request into an ephemeral Codex turn. Codex app-server is an agent protocol rather than a raw model API, so this adapter translates conversations, JSON output, and function calls. Codex app-server and its local authentication integration are experimental. This setup is for a single-user local PentAGI installation: every request to the bridge consumes the host user's Codex plan, regardless of which PentAGI user initiated it. Do not expose it as a hosted or shared service. Run `ctester` on your chosen model and configuration before a real flow. Use PentAGI only against authorized targets.

## Requirements

- PentAGI and a Codex CLI installed on the same host. The CLI must report `Logged in using ChatGPT` from `codex login status`.
- Python 3.10 or newer. The bridge uses only the standard library.
- Docker Desktop or another deployment that can reach the host bridge from the PentAGI container.
- A Codex model available to the signed-in ChatGPT account. The bridge defaults to `gpt-6.1-sol`; pass `--model` to change it and use the same value for `LLM_SERVER_MODEL`.

## Start the bridge on Windows

Run these commands from the repository root in PowerShell:

```powershell
codex login status
$env:CODEX_BRIDGE_SECRET = [guid]::NewGuid().ToString('N')
python examples/codex-bridge/server.py --host 0.0.0.0 --model gpt-6.1-sol
```

Keep the generated `CODEX_BRIDGE_SECRET` value for PentAGI's `.env`. It is a local bridge access secret, not an OpenAI API key or a model billing token. Binding outside loopback requires this secret. Restrict port 8765 to the local Docker network with your firewall; the bridge serves HTTP without TLS. To use the bridge only from the host, omit `--host 0.0.0.0` and it binds to `127.0.0.1`.

Set these values in PentAGI's `.env`:

```dotenv
LLM_SERVER_URL=http://host.docker.internal:8765/v1
LLM_SERVER_KEY=<the CODEX_BRIDGE_SECRET value>
LLM_SERVER_MODEL=gpt-6.1-sol
LLM_SERVER_PROVIDER=
LLM_SERVER_CONFIG_PATH=
EMBEDDING_PROVIDER=none
```

`EMBEDDING_PROVIDER=none` lets PentAGI run without a separate embedding API credential, but disables semantic memory. To retain that feature without an API key, configure a local Ollama embedding model using PentAGI's embedding settings. Do not set `OPEN_AI_KEY`; the Custom provider is the LLM route here.

After starting PentAGI, select the Custom provider for a flow and test it from the container:

```powershell
docker compose up -d
docker exec -it pentagi /opt/pentagi/bin/ctester -type custom -agents simple -groups basic
docker exec -it pentagi /opt/pentagi/bin/ctester -type custom -agents simple_json -groups json
```

On Linux, `host.docker.internal` may need an explicit Docker host-gateway mapping or a different reachable host address. The bridge must run on the host with the Codex sign-in; putting it in a container without that host's Codex credentials will not reuse the session.

## Compatibility and limits

- The bridge accepts `GET /v1/models` and `POST /v1/chat/completions`, including PentAGI function calls, JSON responses, and SSE response format. It buffers a complete Codex turn before sending an SSE result, so partial tokens do not stream live.
- Each request creates a fresh Codex thread and sends PentAGI's full message history. Concurrent requests are capped at four. Plan usage and model eligibility are controlled by the signed-in ChatGPT account.
- Codex app-server does not expose PentAGI's Chat Completions `max_tokens`, `temperature`, or `top_p` controls through this adapter. An output-limit test therefore fails: the bridge reports `stop` when Codex finishes, rather than claiming a length limit it did not enforce.
- The bridge rejects unsupported tool shapes, unexpected Codex tool execution, failed turns, and malformed function arguments. It does not silently fall back to an API key.
- Image and other non-text message parts are rejected. Browser screenshot analysis needs a multimodal adapter before those PentAGI flows can use this bridge.
- Embeddings are separate from LLM inference. The Codex sign-in does not provide an embeddings endpoint.

OpenAI documents [Codex app-server](https://learn.chatgpt.com/docs/app-server) and its [local authentication behavior](https://learn.chatgpt.com/docs/auth). PentAGI's [Custom provider documentation](https://github.com/vxcontrol/pentagi/blob/main/README.md#custom-llm-provider-configuration) describes the Chat Completions interface the bridge supplies.
