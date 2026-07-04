

# 環境構築
```bash
python -m venv /opt/venvs/.venv
source /opt/venvs/.venv/bin/activate
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
source ./activate_tf_gpu.sh
export PROJECT_ROOT_DIR=/workspaces/chabsa-document-classification

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir chabsa-document-classification
cd chabsa-document-classification
git init

```

# gitパスワード入力を省略
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
# 以降、git push などでパスワード入力が不要になる
```

# AI向けドキュメントについて
フォルダ：document にAI指示用のプロンプトを作成
ユースケース毎にファイルを作成しＡＩにコードを作成させる
対話AIが作成できない場合は自作する
距離重み付きONE-HOTのベクトル化関数は対話ＡＩが作成できなかったため自作としている


# 環境メモ
Windowsホスト+devcontainer+GPU 条件で tensorflow + GPU を動作させる
tensorflow + GPU は現在は python3.11~3.12 でしか動作しない、python3.13では動作しない
Linux、WSL等の環境にマッチした cuda ドライバをインストールする必要がある
activate_tf_gpu.sh を実行し、GPU有効にする必要がある
