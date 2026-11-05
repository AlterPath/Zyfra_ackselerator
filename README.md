# Акселератор «Цифра»: кейс 2

[English version](README.en.md)

Техническое решение по кейсу 2 акселератора «Цифра» (октябрь 2024): программное обеспечение квадрокоптера и нейросеть для распознавания действий людей на снимках.

## Программа полёта (`Drone_Software/`)

- `flight.py`: полёт квадрокоптера Clover в симуляторе Gazebo через сервисы `clover` (ROS): взлёт на 1 м, перемещения и повороты по точкам, посадка.
- `camera_sensor.urdf.xacro`: описание камеры для модели дрона.
- `Installation`: пошаговая установка ROS Noetic, симулятора Clover и запуска на Ubuntu 20.04.

## Нейросеть (`Neural_network/`)

`humanrecog.py` распознаёт действия людей на изображениях:
- обучающая и тестовая выборки описаны в `Training_set.csv` и `Testing_set.csv` (имя файла и метка класса);
- изучение данных и распределения классов;
- две модели с переносом обучения на TensorFlow (Keras): EfficientNetB7 и VGG16 с полносвязными слоями сверху. У EfficientNet 40 эпох обучения, метрика у обеих моделей accuracy.

Сами изображения датасета в репозиторий не входят, ссылка на них есть в `Neural_network/Installation`.

## Стек

Python, ROS Noetic, Clover (COEX), Gazebo, TensorFlow (Keras), pandas, Matplotlib, Plotly.

## Лицензия

[MIT](LICENSE)

---

<sub>Код выложен на GitHub в октябре 2024, README добавлен в ноябре 2026.</sub>
