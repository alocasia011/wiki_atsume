#wikipediaのページをランダムに表示し、タイトルと最初の５段落を取得する。
#ある程度ページの情報を集めたらlabel_wikipedia_pages("wikipedia_pages.txt") の部分で面白いと思うページを選んでいく
#集めたページの情報+面白い面白くないの情報を合わせて機械学習モデルを作る。
#面白いページのパターンと面白くないページのパターンを学習させて、機械学習モデルが面白いか面白くないか判断できるようにするというイメージ。
#できたモデルは保存する。作成者は500個のページについて「面白い・面白くない」をラベル付けし、そのうち400個を学習に用いた。（残りはテストに用いた）
#テストでは、機械学習モデルが面白いと判断したもののうち、4割は面白くないものだった。また、面白い記事のうち実際に面白いと判断されたものは8割程度だった。つまり現状はかなりポンコツ。
#直下にあるライブラリはデフォルトで入っていないものも多いので「pip install ホニャララ」とかでinstallする必要がある。

import requests
from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import numpy as np
import MeCab

def get_random_wikipedia_page():
    """ランダムなWikipediaページのURLを取得して、その内容を返す"""
    url = "https://ja.wikipedia.org/wiki/特別:おまかせ表示"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    title = soup.find("h1", id="firstHeading").text
    content_paragraphs = soup.find("div", {"class": "mw-parser-output"}).find_all("p", limit=5)  # 最初の5つの段落を取得

    # 段落のテキストを結合
    content = "\n".join(paragraph.text for paragraph in content_paragraphs if paragraph.text)

    # コンテンツが空の場合は、"内容なし"と表示
    if not content:
        content = "内容なし"

    return title, content

def save_page_content(title, content, file_name):
    """ページの内容をファイルに保存する"""
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(f"タイトル: {title}\n")
        file.write(f"内容: \n{content}\n\n")

def collect_random_pages(num_pages, output_file):
    """指定された数のランダムなWikipediaページを収集してファイルに保存する"""
    for _ in range(num_pages):
        title, content = get_random_wikipedia_page()
        save_page_content(title, content, output_file)
        print(f"Saved: {title}")
        
def label_wikipedia_pages(input_file):
    """Wikipediaページにラベルを付ける関数。各エントリについて一度だけ評価を行う。"""
    entries = []  # 分割されたエントリを保持するリスト
    with open(input_file, "r", encoding="utf-8") as file:
        entry = []  # 現在のエントリの内容を保持
        for line in file:
            if line.startswith("タイトル:") and entry:
                # 新しいエントリの開始を示す。現在のエントリをリストに追加し、新たに開始する
                entries.append("\n".join(entry))
                entry = [line.strip()]  # 現在の行を新しいエントリの最初の行として追加
            else:
                # 現在のエントリに行を追加
                entry.append(line.strip())
        # ファイルの最後のエントリを追加
        entries.append("\n".join(entry))

    labeled_entries = []  # ラベル付けされたエントリを保存するリスト

    for entry in entries:
        print(entry)  # エントリの内容を表示
        label = input("面白い場合は「o」、面白くない場合は「x」を入力してください: ")
        while label not in ["o", "x"]:
            print("無効な入力です。")
            label = input("面白い場合は「o」、面白くない場合は「x」を入力してください: ")
        labeled_entry = f"{entry}\n面白さ: {'interesting' if label == 'o' else 'not interesting'}"
        labeled_entries.append(labeled_entry)

    # ラベル付けされたエントリをファイルに書き戻す
    with open(input_file, "w", encoding="utf-8") as file:
        for entry in labeled_entries:
            file.write(f"{entry}\n\n")

# Janomeのトークナイザーを初期化
t = Tokenizer()

# Janomeを使用してテキストをトークン化する関数
def tokenize(text):
    tokens = t.tokenize(text, wakati=True)  # テキストをトークンのリストに変換
    return tokens

# データの読み込みと前処理
def load_and_preprocess_data(filename,filename2):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.readlines()
    
    texts = []
    labels = []
    text = ""
    for line in content:
        if "面白さ: interesting" in line:
            labels.append(1)  # 面白い
            if text: texts.append(text)
            text = ""  # テキストをリセット
        elif "面白さ: not interesting" in line:
            labels.append(0)  # 面白くない
            if text: texts.append(text)
            text = ""  # テキストをリセット
        else:
            text += line
    with open(filename2, "r", encoding="utf-8") as file:
        content = file.readlines()
        
    text = ""
    for line in content:
        if "面白さ: interesting" in line:
            labels.append(1)  # 面白い
            if text: texts.append(text)
            text = ""  # テキストをリセット
        elif "面白さ: not interesting" in line:
            labels.append(0)  # 面白くない
            if text: texts.append(text)
            text = ""  # テキストをリセット
        else:
            text += line
        
    return texts, labels

# MeCabの初期化
tagger = MeCab.Tagger()

# 固有名詞の濃度を計算（全単語数で割る）
def calculate_proper_nouns_density(text):
    proper_nouns_count = 0
    total_words_count = 0
    node = tagger.parseToNode(text)
    while node:
        # 単語のカウント（BOS/EOSを除く）
        if node.feature.split(",")[0] != "BOS/EOS":
            total_words_count += 1
        # 固有名詞のカウント
        if node.feature.startswith("名詞,固有名詞"):
            proper_nouns_count += 1
        node = node.next
    
    proper_nouns_density = proper_nouns_count / total_words_count if total_words_count > 0 else 0
    return proper_nouns_density


# N個のランダムなページを収集。今は50ページ収集することになっているが下の50という数字を変えると変えられる。
#収集したページの内容は"wikipedia_pages.txt"に保存される。

#ページを集めたいときは１行下のコメントアウトを解除してください
#collect_random_pages(50, "wikipedia_pages.txt")

#label付け（手動で行う。ページの情報が表示されるので面白かったらo、面白くなかったらxを打ち込む）

#ラベルをつけたいときは１行下のコメントアウトを解除してください
#label_wikipedia_pages("wikipedia_pages.txt")

#学習に使うデータを選択（今は二つのtxtファイル 100データと400データを入力にしているが２つである意味はない）
#流れとしては上でラベル付した"wikipedia_pages.txt"を使うことになる。
#"wikipedia_pages3.txt","wikipedia_pages2.txt"を使用するとコード作成者によるラベル付けデータが学習に使われる。
texts, labels = load_and_preprocess_data("wikipedia_pages3.txt","wikipedia_pages2.txt")
# TfidfVectorizerを初期化し、カスタムトークナイザーを使用
vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=1000)

# テキストを特徴ベクトルに変換
features = vectorizer.fit_transform(texts)

# データセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 固有名詞の使用頻度を計算
proper_nouns_counts = np.array([calculate_proper_nouns_density(text) for text in texts]).reshape(-1, 1)

# TF-IDF特徴ベクトルと固有名詞の使用頻度を組み合わせる
#機械学習に用いる特徴量はいろいろ工夫ができる。１つの工夫として固有名詞の濃度を入れてみた。
combined_features = np.hstack((features.toarray(), proper_nouns_counts))

X_train2, X_test2, y_train2, y_test2 = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

#面白いと思うページに出会う確率が基本的に低いので、重み付して面白いと思いそうなページを多少は逃しにくいようにしている。4となっているところを大きくすればするほど逃しにくくなるが全体の精度は下がる。
class_weights = {0: 1, 1: 4}

# ロジスティック回帰モデルの初期化時にクラスの重みを指定
#使うモデルもいろいろ工夫ができる。ハイパーパラメータの調整とかで良くなる場合もある。
model = LogisticRegression(class_weight=class_weights)

# クラスの重み付けを行うロジスティック回帰モデル
#重み付けを特別しない場合の例
#model = LogisticRegression(class_weight='balanced')

# モデルの訓練
case = 2
if(case == 2):
    model.fit(X_train2, y_train2)
elif(case == 1):
    model.fit(X_train, y_train)

# テストデータでの予測
if(case == 2):
    predictions = model.predict(X_test2)
    print(classification_report(y_test2, predictions))
    print("learning set")
    predictions = model.predict(X_train2)
    print(classification_report(y_train2, predictions))
elif(case ==1):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    
#作ったモデルを保存する。
joblib.dump(model, 'find_interesting_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("モデルとベクトルライザーが正常に保存されました。")