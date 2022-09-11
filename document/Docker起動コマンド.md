Dockerのイメージダウンロード && コンテナ起動コマンド

```bash
docker run -p 6080:80 --shm-size=512m tiryoh/ros2-desktop-vnc:galactic
```

上記を実行した後にブラウザで

http://127.0.0.1:6080/

を開くとGUIに入れる。
