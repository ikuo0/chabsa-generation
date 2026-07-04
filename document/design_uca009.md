# 書類ルール
- UCA ユースケースAグループ、親階層
- UCB ユースケースBグループ、子階層
- 要件・仕様は境界が曖昧なので混在で記載してある
- SPEC 仕様、要件の詳細、表や一覧形式で欲しい物は別途記載する


ユースケースは目的を記載
要件、仕様は制限、ルール等を記載
ソースコードは UCB 単位でフォルダを作成、その下に主処理を記載したコードを配置
ファイル実行は、cdせずに起動時（プロジェクトroot）からの実行とする
実装関数とエントリー関数(main)は分けて実装、他処理からの呼び出しを想定する、そのため実装関数はsys.argvなどは使わず引数実装とする

# 要件・仕様

$root = /workspaces/chabsa-document-classification

- UCA009 ＮＮで単語予測モデルを作成する
  - UCB001 単語予測モデルを作成する
    - 概要
      - chABSA データを用いて、単語予測モデルを作成する
      - tf.sparse.SparseTensor による学習を行い、Xデータは粗ベクトル、Yデータは単語インデックス配列とすることでメモリ節約をする（6000次元Xデータに6000次元のONE-HOT答えデータだとメモリをかなり使うため）
    - 実装ファイル
      - $root/src/uca009/ucb001/main.py
    - 実行時オプション
      - reset: 学習済みモデルを削除して再学習する
      - continue: 学習済みモデルがあればそれを読み込んで続きから学習する
        - 1iter毎モデルのうち最新 epoch のモデルを優先して読み込む
        - 学習ログTSVは既存データへ追記する
    - 入力: Xデータ
      - $root/data/uca002/ucb002/chabsa_vector.npz
      - 粗ベクトルのまま利用する
    - 入力: Yデータ
      - $root/data/uca002/ucb002/chabsa_next_word.npy
      - 単語インデックス配列のまま利用する
    - 入力: SentencePieceモデル
      - $root/data/uca001/ucb006/spm_model.model
      - Xデータの次元数を取得する（単語数=次元数）
    - 出力: 学習済みモデル
      - 学習済みモデルは複数の形式で保存しておく、他処理で利用できるようにする
      - $root/data/uca009/ucb001/chabsa_next_word_model.h5
      - $root/data/uca009/ucb001/chabsa_next_word_model.keras
      - 1iter(=1epoch)毎にも保存する
        - $root/data/uca009/ucb001/chabsa_next_word_model_epoch_{epoch}.h5
        - $root/data/uca009/ucb001/chabsa_next_word_model_epoch_{epoch}.keras
      - 出力パスの文字列は関数で取得できるようにする、他処理でも利用する
    - 出力: 学習ログTSV
      - $root/data/uca009/ucb001/chabsa_next_word_train_log.tsv
      - 1iter(=1epoch)毎に loss と sparse_categorical_accuracy を追記する
      - ヘッダ: epoch, loss, sparse_categorical_accuracy
    - 処理:
      - X, Y データを読み込む
        - 次元数は SentencePieceモデルから取得する
        - Xデータを tf.sparse.SparseTensor に変換する(csr_matrix -> tf.sparse.SparseTensor)
        - YデータはINT配列そのままで良いが tf 用に tf.constant に変換する
      - Tensorflow モデルを準備する
        - inputs
          - tf.keras.Input(shape=(次元数,), sparse=True)
        - layers
          - x = tf.keras.layers.Dense(512, activation='relu')(inputs)
          - x = tf.keras.layers.Dense(256, activation='relu')(x)
        - outputs
          - tf.keras.layers.Dense(次元数)(x)
        - model
          - model = tf.keras.Model(inputs=inputs, outputs=outputs)
        - compile
          - optimizer: Adam
          - loss: SparseCategoricalCrossentropy(from_logits=True)
          - metrics: SparseCategoricalAccuracy
        - fit
          - DEFAULT_MAX_ITER = 1000
          - model.fit(X, Y, epochs=DEFAULT_MAX_ITER, verbose=1)
          - 1iter(=1epoch)毎にモデルを保存する
          - 1iter(=1epoch)毎の loss, sparse_categorical_accuracy をTSVへ追記する
          - continue 時は前回の保存 epoch から再開し、DEFAULT_MAX_ITER に達するまで学習する
          - reset 時は過去の学習済みモデル・1iter毎モデル・学習ログTSVを削除して最初から学習する

  - UCB002 作成したモデルで推定をする
    - 実装ファイル
      - $root/src/uca009/ucb002/main.py
    - 実行時オプション
      - none: 最初の単語から順に予測していく、最初の単語は <s> を14個与えて次を予測させる
        - </s> が出るまでもしくは50単語に到達するまで予測を続けて出力する
      - none10: none を10回繰り返す
      - take "文字列": 文字列を与えて次の単語を予測する、文字列は SentencePiece で分割される
        - SentencePiece で分割し、先頭に <s> を14個付与、末尾15単語から次の単語を予測する
        - </s> が出るまでもしくは50単語に到達するまで予測を続けて出力する
    - 入力: SentencePieceモデル
      - $root/data/uca001/ucb006/spm_model.model
      - Xデータの次元数を取得する（単語数=次元数）
    - 入力: 学習モデル
      - $root/data/uca009/ucb001/chabsa_next_word_model.h5
    - 出力: 推定結果
      - $root/data/uca009/ucb002/chabsa_next_word_predict.txt
      - 出力パスの文字列は関数で取得できるようにする、他処理でも利用する
    - 処理の基本的な流れ
      - <s> * 15 + 与えられた文章を分割した単語列を作成
      - 予測ループ
        - 末尾15単語から次の単語を予測する
        - 単語列に予測した単語を追加する
        - </s> が出るまでもしくは50単語に到達するまで予測を続けて出力する
        - 先頭の次の単語予測に戻って再度予測
      - 予測ループで生成した単語インデックス配列を文字化して出力する
        - <s> は出力しない、</s> は出力する
