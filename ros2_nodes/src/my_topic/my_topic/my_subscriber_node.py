import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MySubscriberNode(Node):
    def __init__(self):
        super().__init__("my_subscriber_node")
        self.subscription = self.create_subscription(String, "my_topic", self.on_subscribe, 10)

    def on_subscribe(self, msg):
        self.get_logger().info(f"Subscribe : {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = MySubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
