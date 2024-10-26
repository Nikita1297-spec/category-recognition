FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

ENV PYTHONPATH=/app

COPY ./pyproject.toml /app/pyproject.toml
RUN poetry install --no-root
COPY ./ /app

EXPOSE 5000

CMD ["sh", "/app/service_up.sh"]