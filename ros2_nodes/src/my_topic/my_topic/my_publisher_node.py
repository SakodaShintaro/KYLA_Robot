import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MyPublisherNode(Node):
    def __init__(self):
        super().__init__("my_publisher_node")
        self.publisher = self.create_publisher(String, "my_topic", 10)
        self.timer = self.create_timer(1, self.on_tick)

    def on_tick(self):
        msg = String()
        msg.data = "Hello World!"
        self.publisher.publish(msg)
        self.get_logger().info(f"Publish : {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = MyPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
