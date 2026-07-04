

# 書類ルール
- UCA ユースケースAグループ、親階層
- UCB ユースケースBグループ、子階層
- 要件・仕様は境界が曖昧なので混在で記載してある
- SPEC 仕様、要件の詳細、表や一覧形式で欲しい物は別途記載する


ユースケースは目的を記載
要件、仕様は制限、ルール等を記載
ソースコードは UCB 単位でフォルダを作成、その下に主処理を記載したコードを配置

# 要件・仕様

$root = /workspaces/chabsa-document-classification

- UCA001 chABSAデータを機械学習用に準備する
  - UCB001 chABSAデータを取得する
    - 実装ファイル: $root/src/uca001/ucb001/main.py
    - 実装関数名: get_chabsa_dataset()
    - エントリー関数: main()
    - 実行方法: python3 main.py
    - 書き込み pickle ファイル: $root/data/uca001/ucb001/chabsa_dataset.pkl
      - 作成時に既にファイルがあれば削除する
      - "/data/uca001/ucb001/chabsa_dataset.pkl" 文字列は関数で取得できるようにする、他処理でも利用する
    - ファイル実行時オプション
      - オプション無し: ダウンロード、SQLiteに保存まで実行
      - check_keys: pklファイルのメンバ変数、キーを確認する
  - UCB002 chABSAデータをSQLiteに展開する
    - 実装ファイル: $root/src/uca001/ucb002/main.py
    - 実装関数名: store_sqlite()
    - エントリー関数: main()
    - 実行方法: python3 main.py
    - 入力 pickle ファイル: $root/data/uca001/ucb001/chabsa_dataset.pkl (UCB001 を import して利用する)
    - 書き込み SQLite: $root/data/uca001/ucb002/chabsa_origin.db
      - 作成時に既にファイルがあれば削除する
      - "/data/uca001/ucb002/chabsa_origin.db" 文字列は関数で取得できるようにする、他処理でも利用する
    - SQLiteフィールド: row_id(int), sentence(text), target(text), polarity(text)
    - ファイル実行時オプション
      - オプション無し: pickleファイルを展開してSQLiteに保存まで実行
      - sample: 保存したSQLiteからランダムにサンプルを5件取得して表示する
  - UCB003 文章の正規化を行う
    - 実装ファイル: $root/src/uca001/ucb003/main.py
    - 実装関数名: normalize_text()
    - エントリー関数: main()
    - 実行方法: python3 main.py
    - 入力 SQLite: $root/data/uca001/ucb002/chabsa_origin.db (UCB002 を import して利用する)
    - 書き込み SQLite: $root/data/uca001/ucb003/chabsa_normalized.db
      - 作成時に既にファイルがあれば削除する
      - "/data/uca001/ucb003/chabsa_normalized.db" 文字列は関数で取得できるようにする、他処理でも利用する
    - SQLiteフィールド: row_id(int), sentence(text), target(text), polarity(text)
    - ファイル実行時オプション
      - オプション無し: SQLiteを展開して正規化してSQLiteに保存まで実行
      - sample: 正規化前のDBから3件ランダムに取得、該当する正規化後のデータも取得、正規化前後の文章を 前→後 セットで上下に並べて表示する
  - UCB004 単語数別のトークン化を行う、単語数パラメータ(vocab_size)を複数パターン出力
    - 実装ファイル: $root/src/uca001/ucb004/main.py
    - 実装関数名: tokenize_text()
    - エントリー関数: main()
    - 実行方法: python3 main.py
    - 入力 SQLite: $root/data/uca001/ucb003/chabsa_normalized.db (UCB003 を import して利用する)
    - 書き込み SQLite: $root/data/uca001/ucb004/chabsa_tokenized_{vocab_size}.db
      - 作成時に既にファイルがあれば削除する
      - "/data/uca001/ucb004/chabsa_tokenized_{vocab_size}.db" 文字列は関数で取得できるようにする、他処理でも利用する
    - SentencePieceを使用してトークン化する
      - モデルプレフィクス: $root/data/uca001/ucb004/spm_model_{vocab_size}
    - SQLiteフィールド: vocab_id(int), vocab(text), score(float)
      - score は SentencePiece の piece_score をそのまま保存する
    - vocab_size は 1400開始、指数的に増やして 10000 までの範囲で１０パターン設定
      - n, m, step_count パラメータを引数にとり、n から m まで step_count 回の範囲で指数的に増やす、範囲値は n,m を含む
      - 回数計算・取得はlistを返す関数として実装する
    - ファイル実行時オプション
      - オプション無し: SQLiteを展開してトークン化してSQLiteに保存まで実行
      - sample {絶対ファイルパス} 指定されたトークンDBからランダムにサンプルを20件取得して表示する
        - 絶対ファイルパス は本処理で作成されたトークンDBファイルのパスをコピーしてそのまま貼り付ける
  - UCB005 文章情報の解析を行う、UCB003 で作成したパターンの統計を取る
    - 解析するだけ、最適値は目視で開発者が確認する
    - 実装ファイル: $root/src/uca001/ucb005/main.py
    - 実装関数名: analyze_tokenized_data()
    - エントリー関数: main()
    - 実行方法: python3 main.py
    - 入力 SQLite: $root/data/uca001/ucb004/chabsa_tokenized_{\d+}.db (UCB004 を import して利用する)を列挙
    - 出力 TSV: $root/data/uca001/ucb005/chabsa_tokenized_analyzed.tsv
      - 作成時に既にファイルがあれば削除する
      - "/data/uca001/ucb005/chabsa_tokenized_analyzed.tsv" 文字列は関数で取得できるようにする、他処理でも利用する
    - TSV
      - vocab_size 1種類につき集計１レコード、
      - 項目
        - vocab_size
        - 1文章あたりのトークン文字数平均
        - 1文章あたりのトークン数平均
        - 文章全体の低頻度トークン数（全体を通して出現回数が2回以下のトークンの数）
        - 文章全体の低頻度トークン数割合
        - 1文章あたりのトークン数の分散
        - 1文章あたりのトークン数の標準偏差
        - 1文章あたりのトークン数の最小値
        - 1文章あたりのトークン数の最大値
        - 1文章あたりのトークン数の中央値
        - トークン数／文字数　の全平均
    - ファイル実行時オプション
      - オプション無し: SQLiteを展開して統計を取りTSVに保存まで実行
      - view: 保存したTSVをコンソールにテーブル表示
  - UCB006 「最適単語数」を決めたら再度トークン化
    - 実装ファイル: $root/src/uca001/ucb006/main.py
    - 実装関数名: tokenize_text(vocab_size: int)
    - エントリー関数: main()
    - 実行方法: python3 main.py
    - 入力 SQLite: $root/data/uca001/ucb003/chabsa_normalized.db (UCB003 を import して利用する)
    - 書き込み SQLite: $root/data/uca001/ucb006/chabsa_tokenized.db
      - 作成時に既にファイルがあれば削除する
      - "/data/uca001/ucb006/chabsa_tokenized.db" 文字列は関数で取得できるようにする、他処理でも利用する
    - SentencePieceを使用してトークン化する
      - モデルプレフィクス: $root/data/uca001/ucb006/spm_model
    - SQLiteフィールド: vocab_id(int), vocab(text), score(float)
    - 書き込み 実行時情報: $root/data/uca001/ucb006/chabsa_tokenized_info.txt
      - 作成時に既にファイルがあれば削除する
      - "/data/uca001/ucb006/chabsa_tokenized_info.txt" 文字列は関数で取得できるようにする、他処理でも利用する
      - 内容: vocab_size, 1文章あたりのトークン文字数平均, 1文章あたりのトークン数平均, 文章全体の低頻度トークン数, 文章全体の低頻度トークン数割合, 1文章あたりのトークン数の分散, 1文章あたりのトークン数の標準偏差, 1文章あたりのトークン数の最小値, 1文章あたりのトークン数の最大値, 1文章あたりのトークン数の中央値, トークン数／文字数　の全平均
    - ファイル実行時オプション
      - 数値: 指定された単語数でトークン化してSQLiteに保存まで実行
        - 最低1400、最大値は未定義
  - note: ベクトル化までは行わない
- SPEC001 UCA001 の要件・仕様
  - chABSAデータ は scikit-learn のデータセットとして取得する
  - 取得したデータはSQLiteに保存する
  - 文章の正規化は、全角統一、伸ばし棒の統一等を行う
  - トークン化はSentencePieceを使用する
  - 文字列全般は UTF-8 で保存する
