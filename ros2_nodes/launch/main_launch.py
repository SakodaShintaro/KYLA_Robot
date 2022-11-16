from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 起動したいノードを記述
        Node(
            package='image_publisher',
            executable='image_publisher_node',
            prefix="xterm -e",
        ),
        Node(
            package='face_extractor',
            executable='face_extractor_node',
            prefix="xterm -e",
        ),
        Node(
            package='face_identifier',
            executable='face_identifier',
            prefix="xterm -e",
        ),
    ])
