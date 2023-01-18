set -eux
colcon build
bash ./install/setup.bash
ros2 launch launch/main_launch.py
