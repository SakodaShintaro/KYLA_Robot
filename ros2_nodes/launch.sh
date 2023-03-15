set -eux
colcon build
source ./install/setup.bash
ros2 launch launch/main_launch.py
