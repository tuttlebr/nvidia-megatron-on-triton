ARG FROM_BASE_IMAGE
FROM ${FROM_BASE_IMAGE}
WORKDIR /workspace
RUN pip3 install tritonclient[all] transformers jupyter jupyterlab