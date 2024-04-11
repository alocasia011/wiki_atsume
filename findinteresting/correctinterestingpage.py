#モデルが最強であれば、面白いwikipediaのページがinteresting_articles.txtにじゃんじゃんたまることになる。
#現状、面白ページを見逃すことも多いし、面白くないページを収集することのほうが多い気がする。
import requests
from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
import joblib # モデルを保存/読み込みするために使用
import numpy as np
import MeCab

tagger = MeCab.Tagger()


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
    
    # 固有名詞の濃度を計算（全単語数で割る）
    proper_nouns_density = proper_nouns_count / total_words_count if total_words_count > 0 else 0
    return proper_nouns_density


# Wikipediaの記事をランダムに取得する関数
def get_random_wikipedia_page():
    """ランダムなWikipediaページのURLを取得して、その内容を返す"""
    url = "https://ja.wikipedia.org/wiki/特別:おまかせ表示"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    title = soup.find("h1", id="firstHeading").text
    #print(title)
    content_paragraphs = soup.find("div", {"class": "mw-parser-output"}).find_all("p", limit=5)
    content = "\n".join(paragraph.text for paragraph in content_paragraphs if paragraph.text)
    if not content:
        content = "内容なし"
    return title, content, response.url

# Janomeのトークナイザーを初期化
t = Tokenizer()

# Janomeを使用してテキストをトークン化する関数
def tokenize(text):
    tokens = t.tokenize(text, wakati=True)
    return tokens

def predict_and_save_interesting_articles():
    title, content, url = get_random_wikipedia_page()
    tokens = tokenize(content)
    text_features = vectorizer.transform([' '.join(tokens)])
    
    # 固有名詞の使用頻度を計算
    proper_nouns_density = calculate_proper_nouns_density(content)
    # proper_nouns_densityを2次元配列に変換
    proper_nouns_density_array = np.array([[proper_nouns_density]])
    
    # TF-IDF特徴ベクトルと固有名詞の使用頻度を組み合わせる
    combined_features = np.hstack((text_features.toarray(), proper_nouns_density_array))
    
    prediction = model.predict(combined_features)
    
    if prediction == 1:
        print(f"面白い!: {title}")
        with open("interesting_articles.txt", "a", encoding="utf-8") as file:
            file.write(f"タイトル: {title}\nURL: {url}\n\n")
        dt = 1
    else:
        print(f"うーん: {title}")
        dt = 0
    return dt

# FindInterestingモデルとTfidfVectorizerのロード
model = joblib.load('find_interesting_model.pkl') # モデルファイルの名前は例としています
vectorizer = joblib.load('tfidf_vectorizer.pkl') # 同様に、適切なファイル名を使用してください


# モデルを使ってランダムなWikipediaの記事が面白いか判断し、結果に応じて保存
i=0
#何個面白い記事を見つけたいか。実際のモデルの精度は6割とかなので、6個面白いページを見つけるためには
#「モデルが面白いと判断したページ」は10個くらい必要になる。
#これは作成者の場合で、実際に使うときはモデル作成時の値を見る必要がある。
N = 3
while(i <N):
    i += predict_and_save_interesting_articles()
    #i +=1
print("面白いことを集めたよ！")