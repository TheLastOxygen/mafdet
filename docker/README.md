# Open WebUI + MAFDET in Docker

`docker-compose.yml` runs three containers: llama.cpp's server with Qwen3-4B, the MAFDET proxy in
front of it, and Open WebUI talking to the proxy.

```bash
cd docker
docker compose up -d
```

- Open WebUI: http://localhost:3000 (create the admin account the first time).
- The proxy's status: http://localhost:8077/proxy.
- The first start downloads the model (~2.5 GB) into the `models` volume; the proxy waits for
  llama-server before it starts (`docker compose logs -f mafdet`).

Then install the user script in the browser, as in
[Open WebUI + MAFDET in 5 minutes](../mafdet/proxy/clients/README.md), step 4: it already matches
`http://localhost:3000` and sends the drafts to `http://127.0.0.1:8077`, the port the compose file
publishes.

## Choices

- **Another model**: change `-hf Qwen/Qwen3-4B-GGUF:Q4_K_M --alias qwen3-4b` (any GGUF repository and
  quantization on Hugging Face), and `--model` in the `mafdet` service to the same alias.
- **Thinking ahead** (Qwen3, thinking on): add `--think-ahead` to the `mafdet` command.
- **No NVIDIA GPU**: use the image `ghcr.io/ggml-org/llama.cpp:server` (CPU), remove `-ngl 99` and the
  `deploy` block. For an AMD or Intel GPU on Linux: `ghcr.io/ggml-org/llama.cpp:server-vulkan`, no
  `deploy` block, and `devices: ["/dev/dri:/dev/dri"]` on the `llama` service.
- **Your own llama-server** instead of the container: remove the `llama` service and point the proxy at
  yours (`--url http://host.docker.internal:8081` on Docker Desktop). Start it with
  `-np 2 -kvu --no-cache-idle-slots` and a context of at least 16k.
- **Only the proxy's image**: `docker build -f docker/Dockerfile -t mafdet-proxy .` from the
  repository's root.

## Status

Written to the same options as the setup measured on Windows (`research/LOG.md`, 2026-11-06), but not
yet run in Docker: if something does not work, an issue with `docker compose logs` helps.
