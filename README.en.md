# Tsifra Accelerator: Case 2

[Русская версия](README.md)

A technical solution for case 2 of the Tsifra digital accelerator (October 2024): quadcopter flight software and a neural network that recognizes human actions in images.

## Flight program (`Drone_Software/`)

- `flight.py`: a Clover quadcopter flight in the Gazebo simulator through the `clover` ROS services: takeoff to 1 m, moves and turns through waypoints, landing.
- `camera_sensor.urdf.xacro`: the camera description for the drone model.
- `Installation`: step-by-step setup of ROS Noetic, the Clover simulator and the launch on Ubuntu 20.04.

## Neural network (`Neural_network/`)

`humanrecog.py` recognizes human actions in images:
- the training and test sets are listed in `Training_set.csv` and `Testing_set.csv` (file name and class label);
- data exploration and class distribution;
- two transfer-learning models in TensorFlow (Keras): EfficientNetB7 and VGG16 with dense layers on top. EfficientNet trains for 40 epochs, and both models are evaluated by accuracy.

The dataset images are not included; the link to them is in `Neural_network/Installation`.

Instructions and comments are in Russian.

## Tech stack

Python, ROS Noetic, Clover (COEX), Gazebo, TensorFlow (Keras), pandas, Matplotlib, Plotly.

## License

[MIT](LICENSE)
