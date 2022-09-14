import rclpy
from rclpy.node import Node


def main():
    # RCLの初期化
    rclpy.init()

    # ノードの生成
    node = Node("hello_node")

    # ログ出力
    node.get_logger().info("Hello World!")

    # ノード終了の待機
    rclpy.spin(node)

    # ノードの破棄
    node.destroy_node()

    # RCLのシャットダウン
    rclpy.shutdown()


if __name__ == "__main__":
    main()
