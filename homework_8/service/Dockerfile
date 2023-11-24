FROM python:3.11

WORKDIR /code

RUN pip install poetry

COPY ./pyproject.toml /code/

RUN poetry install --no-root --no-dev

COPY ./src /code/src

COPY ./data/models/quantized /code/models/uk_model/

CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
