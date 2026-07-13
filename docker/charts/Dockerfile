FROM python:3.13.1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN mkdir /experiment
WORKDIR /experiment
COPY .python-version ./
COPY process.py ./
COPY pyproject.toml uv.lock ./
RUN apt-get update
RUN apt install -y texlive-latex-extra texlive-fonts-recommended dvipng cm-super
RUN uv sync --locked
CMD uv run process.py
