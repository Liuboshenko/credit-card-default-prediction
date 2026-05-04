
## Строим докер образ на основании нашего Dockerfile, файл находится в текущей директории проекта 
#### `docker build -t liuboshenko/credit-card-ml-service:latest_v1 .`

```bash
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ docker build -t liuboshenko/credit-card-ml-service:latest_v1 .

[+] Building 0.7s (17/17) FINISHED  docker:default
 => [internal] load build definition from Dockerfile                          0.0s
 => => transferring dockerfile: 2.05kB                                        0.0s
 => [internal] load metadata for docker.io/library/python:3.11-slim            0.6s
 => [internal] load .dockerignore                                              0.0s
 => => transferring context: 2B                                                0.0s
 => [internal] load build context                                              0.0s
 => => transferring context: 1.13kB                                            0.0s
 => [builder 1/4] FROM docker.io/library/python:3.11-slim@sha256:...           0.0s
 => CACHED [builder 2/4] WORKDIR /app                                          0.0s
 => CACHED [builder 3/4] COPY requirements.txt .                               0.0s
 => CACHED [builder 4/4] RUN pip install --no-cache-dir --upgrade pip \
                        && pip install --no-cache-dir -r requirements.txt      0.0s
 => CACHED [stage-1  3/10] COPY --from=builder .../site-packages               0.0s
 => CACHED [stage-1  4/10] COPY --from=builder /usr/local/bin                  0.0s
 => CACHED [stage-1  5/10] COPY app/      ./app/                               0.0s
 => CACHED [stage-1  6/10] COPY src/      ./src/                               0.0s
 => CACHED [stage-1  7/10] COPY models/   ./models/                            0.0s
 => CACHED [stage-1  8/10] COPY config.py .                                    0.0s
 => CACHED [stage-1  9/10] COPY wsgi.py   .                                    0.0s
 => CACHED [stage-1 10/10] RUN mkdir -p logs \
                        && adduser --disabled-password --gecos "" appuser \
                        && chown -R appuser:appuser /app                      0.0s
 => exporting to image                                                         0.0s
 => => exporting layers                                                        0.0s
 => => writing image sha256:e7def69ee307a752bda848edbfea7bf98f0abe9f...        0.0s
 => => naming to docker.io/liuboshenko/credit-card-ml-service:latest_v1        0.0s
```
## Пушим образ в публичный репозиторой hub.docker.com `docker push liuboshenko/credit-card-ml-service:latest_v1`
```bash
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ docker push liuboshenko/credit-card-ml-service:latest_v1
The push refers to repository [docker.io/liuboshenko/credit-card-ml-service]
42b3d1a3a3a2: Pushed 
734b2b6f7496: Pushed 
ab61bebdd7cc: Pushed 
1b329c591eb9: Pushed 
1c39e8b191cb: Pushed 
21f2defa12d6: Pushed 
e9d2ab96a5ab: Pushed 
37813dc2782d: Pushed 
fb4ec9bfdd55: Pushed 
2360b28b4660: Mounted from library/python 
248c30140986: Mounted from library/python 
fa1aec823035: Mounted from library/python 
6d7c150df58d: Mounted from library/python 
latest_v1: digest: sha256:dd90e5ac32de271a32f84dceef2c7e53b3697e9f6123a540b7a6a248511be6c3 size: 3037
```
## Проверим что образ отображается в UI удаленного репоизтория
![скрин рабочего приложения](images_screenshots/docker_public_hub.png)

## Для демонстрации убедимся что локально нет образов в хранилище
```bash
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ docker images
REPOSITORY   TAG       IMAGE ID   CREATED   SIZE
```

## Загрузим образ из удаленного репозитория hub.docker.com
```bash
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ docker pull liuboshenko/credit-card-ml-service:latest_v1
latest_v1: Pulling from liuboshenko/credit-card-ml-service
3531af2bc2a9: Already exists 
91ff8760033c: Already exists 
f3ba2250c524: Already exists 
7ccd73948dde: Already exists 
ba4b31c6f0b7: Already exists 
8b4283df1180: Already exists 
a9384f1e877b: Already exists 
2f89eb63da28: Already exists 
e2d6b69f2a44: Already exists 
0d7c88ecce40: Already exists 
dc795677272e: Already exists 
6cfed9a204ba: Already exists 
de6159ea5a26: Already exists 
Digest: sha256:dd90e5ac32de271a32f84dceef2c7e53b3697e9f6123a540b7a6a248511be6c3
Status: Downloaded newer image for liuboshenko/credit-card-ml-service:latest_v1
docker.io/liuboshenko/credit-card-ml-service:latest_v1
```
## Убедимся что образ загружен корректно
```bash
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ docker images
REPOSITORY                           TAG         IMAGE ID       CREATED      SIZE
liuboshenko/credit-card-ml-service   latest_v1   e7def69ee307   3 days ago   499MB
``` 
## Запустим контейнер и проверим что наше приложение работает
```bash
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ docker run -d \
>   --name credit_ml \
>   -p 5000:5000 \
>   liuboshenko/credit-card-ml-service:latest_v1
45d53ba5df4662d729619b7f700f76df66d1a63f0870b4f12e331ab6df483e90
```
## Смотрим что контейнер запущен
```bash
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ docker ps
CONTAINER ID   IMAGE                                          COMMAND                  CREATED          STATUS                    PORTS                                         NAMES
45d53ba5df46   liuboshenko/credit-card-ml-service:latest_v1   "gunicorn --bind 0.0…"   13 seconds ago   Up 12 seconds (healthy)   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp   credit_ml
```
## Дерним эндпоинт curl-ом
```bash
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ curl http://localhost:5000/health
{"models":["v1","v2"],"service":"credit-card-default-prediction","status":"healthy","timestamp":"2026-05-03T18:35:26.975451Z"}
sovereign@sovereign-pc:~/Documents/НИЯУ_МИФИ/INTEGRATION/credit-card-default-prediction$ 
```
## Скрин рабочего приложения 
![скрин рабочего приложения](images_screenshots/available_port_5000.png)