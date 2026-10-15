FROM python:3.14-slim

# if you forked EasyOCR, you can pass in your own GitHub username to use your fork
# i.e. gh_username=myname
ARG gh_username=JaidedAI
ARG service_home="/home/EasyOCR"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Configure apt and install packages (libglib2.0-0 is what opencv-python-headless needs)
RUN apt-get update -y && \
    apt-get install -y \
        libglib2.0-0 \
        git \
        # cleanup
        && apt-get autoremove -y \
        && apt-get clean -y \
        && rm -rf /var/lib/apt/lists

# Clone EasyOCR repo
RUN mkdir "$service_home" \
    && git clone "https://github.com/$gh_username/EasyOCR.git" "$service_home" \
    && cd "$service_home" \
    && git remote add upstream "https://github.com/JaidedAI/EasyOCR.git" \
    && git pull upstream master

# Install with uv; uses the committed uv.lock when present
RUN cd "$service_home" && uv sync --no-dev --compile-bytecode

WORKDIR "$service_home"
