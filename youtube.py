from googleapiclient.discovery import build
import pandas as pd
import streamlit as st
import json

with open("secret.json") as f:
    secret = json.load(f)

DEVELOPER_KEY = secret["KEY"]
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


###  APIの認証部分  ###
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey = DEVELOPER_KEY)

### APIの定義とAPIを叩く(.execute())   ###
def video_search(youtube, q, max_results):

    response = youtube.search().list(
    q = q,
    part = "id, snippet",# (string表示で) partパラメータは、API レスポンスに含める 1 つ以上の search リソース プロパティをカンマ区切りのリストで指定します。パラメータ値を snippet に設定します。
    order = "viewCount",
    type = "video",
    maxResults = max_results
    ).execute()

    items_id = []
    items = response["items"]
    for item in items:
        item_id = {}#辞書型
        item_id["video_id"] = item["id"]["videoId"] #item_idの辞書のkeyとして[video_id]を入れる。/item[A][B]=Aの中にあるB
        item_id["channel_id"] = item["snippet"]["channelId"]
        items_id.append(item_id)

    df_video = pd.DataFrame(items_id)
    return df_video

def get_results(df_video, threshold=5000):

    channel_ids = df_video["channel_id"].unique().tolist() 

    subscriber_list = youtube.channels().list(
        id = ",".join(channel_ids), #channel_idsのリストの要素をカンマで結合する。
        part = "statistics",
        fields = "items(id, statistics(subscriberCount))" 
        #partだけで変数を"statistics"のみにするとその中の"statistics.subscriberCount"以外の情報もとってきてしまう。
    ).execute()

    subscribers = []
    for item in subscriber_list["items"][:5]:
        subscriber = {} #辞書型
        if len(item["statistics"]) > 0:
            subscriber["channel_id"] = item["id"]
            subscriber["subscriber_count"] = int(item["statistics"]["subscriberCount"])#intの理由：後でdf["subscriber_count"]<500とすると文字列は不等式を立てられないので整数値にする
        else:
            subscriber["channel_id"] = item["id"]
        subscribers.append(subscriber)

    df_subscribers = pd.DataFrame(subscribers)

    df = pd.merge(left=df_video, right=df_subscribers, on="channel_id")
    df_extracted = df[df["subscriber_count"] < threshold]

    video_ids = df_extracted["video_id"].tolist()
    videos_list = youtube.videos().list(
        id = ",".join(video_ids),
        part = "snippet, statistics",
        fields = "items(id, snippet(title), statistics(viewCount))"
    ).execute()

    videos_info = []
    items = videos_list.get("items", [])
    for item in items:
        if "statistics" in item and "viewCount" in item["statistics"]:
            video_info = {}
            video_info["video_id"] = item["id"]
            video_info["title"] = item["snippet"]["title"]
            video_info["view_count"] = int(item["statistics"]["viewCount"])
            videos_info.append(video_info)

    df_videos_info = pd.DataFrame(videos_info)

    if df_videos_info.empty:
        st.warning("動画情報の取得に失敗しました。対象の動画が存在しないか、取得に失敗しています。")
        return pd.DataFrame()

    results = pd.merge(left=df_extracted, right=df_videos_info, on="video_id")

    results = results.loc[:,["video_id", "title", "view_count", "subscriber_count", "channel_id"]]
    return results


### アプリ作成  ###
st.title("YouTube分析")

st.sidebar.write("## クエリと閾値の設定")

st.sidebar.write("### クエリの入力")
query = st.sidebar.text_input("検索クエリを入力してください。","vaundy") # 最後のカンマはでデフォの言葉

st.sidebar.write("### 閾値の設定")
threshold = st.sidebar.slider("登録者の閾値", 10, 20000, 5000) # 最後のカンマはでデフォの数値

st.write("### 選択中のパラメータ")
st.markdown(f"""
- 検索クエリ：{query}
- 登録者数の閾値：{threshold}
""")

df_video = video_search(youtube, q=query, max_results=10)

results = get_results(df_video, threshold=threshold)


st.write("### 分析結果", results)

st.write("### 動画再生")

video_id = st.text_input("動画IDを入力してください")
url = f"http://youtu.be/{video_id}"

video_field = st.empty()
video_field.write("こちらに動画が表示されます")

if st.button("ビデオ表示"):
    if len(video_id) > 0:
        try:
            video_field.video(url)
        except:
            st.error("おっと！何かエラーが起きているようです！")
