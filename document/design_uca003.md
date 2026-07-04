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

- UCA003 ベクトル化された chABSA データを圧縮する
  - UCB001 TruncatedSVD圧縮の最適値を見つける
    - 入力: Xデータ
      - $root/data/uca002/ucb002/chabsa_vector.npz
    - 処理:
      - 元の次元数から２００次元ずつ減らしてレポートを作成する
      - １０００次元を下回るまで全てのパターンを実行する
      - 処理はスレッド化し、マルチタスクで実行する
      - 処理１回毎にレポートを作成し返却
        - 列
          - 次元数、圧縮後の次元数、圧縮率、再現率、情報損失率、しきい値達成フラグ、９５％以上フラグ、特異値の合計、最大特異値
          - original_dim, reduced_dim, compression_ratio, reconstruction_rate, information_loss_rate, threshold_ok, is_min_dim_over_threshold, singular_value_sum, top_singular_value
            - 計算式
              - compression_ratio = reduced_dim / original_dim
              - reconstruction_rate = cumulative_explained_variance_ratio
              - information_loss_rate = 1 - reconstruction_rate
              - threshold_ok = reconstruction_rate >= 0.95
              - is_min_dim_over_threshold = threshold_ok が true となる最小 reduced_dim の行だけ true
              - singular_value_sum = sum(singular_values_for_reduced_dim)
              - top_singular_value = max(singular_values_for_reduced_dim)
          - 小数点は 4 桁までとする
    - 出力: TruncatedSVDレポート
      TruncatedSVD処理の結果をレポートとしてTSV出力する
      - $root/data/uca003/ucb001/truncated_svd_report.tsv
    - 出力: TruncatedSVDモデル
      TruncatedSVD処理の結果をモデルとして保存する、実験した全ての圧縮後次元数のモデルを保存する
      - $root/data/uca003/ucb001/truncated_svd_model_{圧縮後次元数}.pkl
      - "/data/uca003/ucb001/truncated_svd_model_{圧縮後次元数}.pkl" 文字列は関数で取得できるようにする、他処理でも利用する、圧縮後次元数 を引数とする

  - UCB002 PCA圧縮しレポートを作成する
    - 実装ファイル
      - $root/src/uca003/ucb002/main.py
    - 実行時オプション
      - 数値: 指定された次元数に圧縮する、最小、最大チェックは行わない
    - 入力: Xデータ
      - $root/data/uca002/ucb002/chabsa_vector.npz
      - 粗ベクトルで保存されているため、読み出し後に密ベクトルへ変換する必要がある
    - 処理:
      - 指定された次元数でPCA圧縮を行い、レポートを作成する
      - レポートについて
        - 列
          - 次元数、圧縮後の次元数、圧縮率、再現率、情報損失率、しきい値達成フラグ、９５％以上フラグ、特異値の合計、最大特異値
          - original_dim, reduced_dim, compression_ratio, reconstruction_rate, information_loss_rate, threshold_ok, is_min_dim_over_threshold, singular_value_sum, top_singular_value
            - 計算式
              - compression_ratio = reduced_dim / original_dim
              - reconstruction_rate = cumulative_explained_variance_ratio
              - information_loss_rate = 1 - reconstruction_rate
              - threshold_ok = reconstruction_rate >= 0.95
              - is_min_dim_over_threshold = threshold_ok が true となる最小 reduced_dim の行だけ true
              - singular_value_sum = sum(singular_values_for_reduced_dim)
              - top_singular_value = max(singular_values_for_reduced_dim)
          - 小数点は 4 桁までとする
      - 出力: PCAレポート
        PCA処理の結果をレポートとしてTSV出力する
        - $root/data/uca003/ucb002/pca_report_{圧縮後次元数}.tsv
        - 1レコードしかないので、0列目に項目、1列目に値を出力する
      - 出力: PCAモデル
        PCA処理の結果をモデルとして保存する、実験した全ての圧縮後次元数のモデルを保存する
        - $root/data/uca003/ucb002/pca_model_{圧縮後次元数}.pkl
        - "/data/uca003/ucb002/pca_model_{圧縮後次元数}.pkl" 文字列は関数で取得できるようにする、他処理でも利用する、圧縮後次元数 を引数とする
