# О проекте

Распознавание цифр из MNIST

## Train
Запустить сервер mlflow.

    mlflow ui -p 8888

Пользователь выбирает порт, он должен быть указан в конфиге. По дефолту -p 8888.

Далее запускаем:

    python mlops_project/train.py

Параметры для этапа train указаны в конфиге. При отсутсвии данных подгружаются через dvc.

## Inference

Запуск сервера:

    python mlops_project/run_server.py --port=8890

Выбор порта за пользователем. После получаем адрес, по которому можно работать с моделью:

    Listening at: http://127.0.0.1:8890 (76870)

С этим адресом нужно запустить test_server

    python mlops_project/test_server.py --servind_addr=http://127.0.0.1:8890

/invocations добавляется само, так что это не нужно делать при запуске.

# Triton

## Как выбирались параметры

Параметры:

```
    instance_group.count
    dynamic_batching.max_queue_delay_microseconds
    dynamic_batching.preferred_batch_size
```

У меня нет гпу, для сервера можно выделить только 5 ядер, поэтому instance_group.count = 1 показал себя лучше. Параметр max_queue_delay равный 2000 показал меньшую задержку в сравнении с max_queue_delay=1000

- MacOS 12.5
- Intel Core i7
- Задача распознавания цифр в MNIST, модель принимает картинки и возвращает вектор из 10 координат
- model_repository:

```
    model_repository
    └── onnx-model
        ├── 1
        │   └── model.onnx
        └── config.pbtxt
```

- метрики throughput и latency

| count | max_queue_delay | throughput (infer/sec)| latency (usec)|
|-------|-----------------|------------|---------|
|   1   |       1000      | 8906.24      |   2252      |
|   1   |       2000      | 10571.6    |   1896    |
|    2   |         1000        |      9742.7      |     2057    |
|    2   |        2000         |      10081.3      |     1987    |

## Конвертация модели
    python MLOps_project/convert_model.py ./models/best_model.xyz ./model_repository/onnx-model/1/model.onnx

## Сервер

Для первого запуска нужно загрузить модель. Далее запуск сервера

    dvc pull model_repository/onnx-model/1/model.onnx.dvc
    docker compose up

Запуск клиента делается следующей командой:

    python MLOps_project/client_triton.py

## Другое

docker run:

    docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk

perf_analyzer:

    perf_analyzer -m onnx-model -u localhost:8500 --concurrency-range 20:20 --shape IMAGES:2,1,28,28
