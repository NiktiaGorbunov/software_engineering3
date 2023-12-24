# Разработка приложения для замены фона у фотографии с человеком

**Цель**: разработать API приложение машинного обучения и развернуть его в облаке. 

**Состав команды**:

1. Горбунов Никита Денисович РИМ-130907
2. Менякин Иван Сергеевич РИМ-130907
3. Родионов Василий Сергеевич РИМ-130907
4. Соломеин Александр Александрович РИМ-130908

## Развертывание модели

**1. Установите Docker**
Установка Docker:
Убедитесь, что на вашем компьютере установлен Docker. Если нет, следуйте инструкциям на официальном сайте Docker: https://docs.docker.com/get-docker/

**2. Склонируйте репозиторий**
```
git clone https://github.com/NiktiaGorbunov/software_engineering3.git
cd repository
```
**3. Cборка образа**
```
docker build -t my-image .
```
**4. Запуск контейнера**
```
docker run -p 8501:8501 my-image
```
Эта команда запускает контейнер и пробрасывает порт 8501, на котором работает ваше приложение. После запуска приложение будет доступно по адресу http://localhost:8501

**5. Использование postman для отправки запросов**

Установите Postman, если у вас его нет: https://www.postman.com/downloads/. Также можно воспользоваться web версией: https://www.postman.com/
Откройте Postman.
Создайте новый запрос.
Укажите метод POST и URL http://194.87.35.95:8501/file/process-files.
В разделе "Body" выберите "form-data".
Добавьте два ключа:
image - прикрепите изображение, которое вы хотите обработать.
background - прикрепите изображение фона.
Нажмите "Send" для отправки запроса.

**6. Получение результата**

После отправки запроса вы получите ответ в формате JSON. В ответе будет содержаться ключ "result", в котором будет содержаться изображение с обработанным фоном.
Вы можете сохранить это изображение или использовать в соответствии с вашими потребностями.

Теперь вы успешно установили, запустили и использовали ваше приложение для обработки изображений с использованием Docker и FastAPI. Если у вас возникнут вопросы или проблемы, обратитесь к документации или к участникам команды для поддержки!

## Модель [MediaPipe Selfie Segmentation](https://chuoling.github.io/mediapipe/solutions/selfie_segmentation.html#mediapipe-selfie-segmentation)
### Обзор
Сегментация селфи MediaPipe позволяет выделить выдающихся людей в кадре. Она может выполняться в режиме реального времени как на смартфонах, так и на ноутбуках. Предполагаемые варианты использования включают эффекты селфи и видеоконференции, при которых человек находится близко (<2 м) к камере

## Реализация API 
[**Fast API**](https://fastapi.tiangolo.com/)

FastAPI - это современный, быстрый (высокопроизводительный) веб-фреймворк для создания API с Python 3.8+ на основе стандартных подсказок по типу Python.

Документация: https://fastapi.tiangolo.com

Исходный код: https://github.com/tiangolo/fastapi




