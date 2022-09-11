## ROS2についてのメモ
ROS2には

* トピック
* サービス
* アクション

という3種類のデータ通信の方式がある。

参考文献) 布留川 英一『Unityではじめる ROS・人工知能 ロボットプログラミング実践入門』

https://www.borndigital.co.jp/book/26553.html

## ワークスペースの作り方
```bash
cd ros2_nodes
mkdir -p ./src
colcon build
source ./install/setup.bash
echo "source $(readlink -f ./install/setup.bash)" >> ~/.bashrc
```

## パッケージの作り方
```bash
cd ./src
ros2 pkg create --build-type ament_python hello --dependencies rclpy
# ./hello/setup.py の entry_pointsを編集
# hello/hello/hello_node.pyを作成
cd ../
colcon build

# 以下を実行するコンソールで実行
source ~/.bashrc
ros2 run hello hello_node
```