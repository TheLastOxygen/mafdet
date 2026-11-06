# The MAFDET proxy alone: it reads ahead for a llama-server running in another container.
# The proxy uses the standard library only, so the package goes in without its dependencies
# (those are for the app: the window, the terminal, the tokenizers).
#
#   docker build -f docker/Dockerfile -t mafdet-proxy .        (from the repository's root)
#
# docker/docker-compose.yml runs it between llama-server and Open WebUI.
FROM python:3.12-slim

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY mafdet ./mafdet
RUN pip install --no-cache-dir --no-deps . && rm -rf /src

# Where the proxy keeps the engine's profile and the last request (the tools, the template options).
ENV MAFDET_HOME=/data
VOLUME /data
EXPOSE 8077

ENTRYPOINT ["python", "-m", "mafdet.proxy"]
# --url: the llama-server container; --host 0.0.0.0: reachable from the other containers and the host;
# --cors: the page that sends the drafts (Open WebUI as published by the compose file).
CMD ["--url", "http://llama:8080", "--model", "qwen3-4b", "--host", "0.0.0.0", \
     "--cors", "http://localhost:3000,http://127.0.0.1:3000"]
