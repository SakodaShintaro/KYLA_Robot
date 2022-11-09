Dockerのイメージダウンロード && コンテナ起動コマンド

```bash
docker run --gpus=all -p 6080:80 --shm-size=512m tiryoh/ros2-desktop-vnc:galactic
```

上記を実行した後にブラウザで

http://127.0.0.1:6080/

を開くとGUIに入れる。

## コンテナ生成後の環境導入など
pipを入れる
```bash
sudo apt install python3-pip
pip3 install -r requirements.txt
```

```bash
sudo apt install xterm -y
```
