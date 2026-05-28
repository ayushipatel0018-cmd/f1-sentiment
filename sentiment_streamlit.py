import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
from transformers import pipeline

st.title("🏎💬 F1 Fan Sentiment Analyser")
st.write("Analysing real fan emotions across drivers — powered by AI")

# ── LOAD MODEL ──
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

analyser = load_model()

# ── DEFAULT DATASET ──
default_comments = [
    {"driver": "Verstappen", "comment": "Max is just untouchable this season, nobody comes close"},
    {"driver": "Verstappen", "comment": "Verstappen is so boring to watch, he just wins everything"},
    {"driver": "Verstappen", "comment": "That overtake from Max was absolutely insane, pure genius"},
    {"driver": "Verstappen", "comment": "I'm so tired of Verstappen winning every single race"},
    {"driver": "Verstappen", "comment": "Max drove brilliantly under pressure today, deserved winner"},
    {"driver": "Verstappen", "comment": "Verstappen got lucky again, that safety car saved him"},
    {"driver": "Verstappen", "comment": "The way Max manages tyres is just on another level"},
    {"driver": "Verstappen", "comment": "Verstappen is the most overrated champion in F1 history"},
    {"driver": "Verstappen", "comment": "Max in the wet is genuinely terrifying, nobody touches him"},
    {"driver": "Verstappen", "comment": "I can't stand Verstappen's attitude, talented but arrogant"},
    {"driver": "Hamilton", "comment": "Lewis is still the greatest of all time, nobody comes close"},
    {"driver": "Hamilton", "comment": "Hamilton should have retired years ago honestly"},
    {"driver": "Hamilton", "comment": "That drive from Lewis today was absolutely masterful"},
    {"driver": "Hamilton", "comment": "I can't believe Hamilton lost that championship in Abu Dhabi"},
    {"driver": "Hamilton", "comment": "Lewis moving to Ferrari is the most exciting thing in years"},
    {"driver": "Hamilton", "comment": "Hamilton keeps making excuses, just accept you lost"},
    {"driver": "Hamilton", "comment": "Seven world titles and still hungry, Lewis is incredible"},
    {"driver": "Hamilton", "comment": "I feel so sad watching Hamilton struggle this season"},
    {"driver": "Hamilton", "comment": "Lewis deserved so much better from Mercedes this year"},
    {"driver": "Hamilton", "comment": "Hamilton at Ferrari is going to be something special"},
    {"driver": "Norris", "comment": "Lando is finally showing he can be a real championship contender"},
    {"driver": "Norris", "comment": "Norris keeps throwing away points with stupid mistakes"},
    {"driver": "Norris", "comment": "That Norris lap in qualifying was genuinely breathtaking"},
    {"driver": "Norris", "comment": "I feel so bad for Lando, he deserved that win so much"},
    {"driver": "Norris", "comment": "Norris is the most naturally talented driver on the grid right now"},
    {"driver": "Norris", "comment": "Lando needs to be more aggressive if he wants to win titles"},
    {"driver": "Norris", "comment": "The way Norris fights back after setbacks is genuinely inspiring"},
    {"driver": "Norris", "comment": "Norris had the pace today but the strategy ruined everything"},
    {"driver": "Norris", "comment": "Lando is going to be world champion one day I genuinely believe it"},
    {"driver": "Norris", "comment": "Norris is too nice, he needs a killer instinct to beat Max"},
    {"driver": "Leclerc", "comment": "Charles is the most naturally gifted driver Ferrari has had in years"},
    {"driver": "Leclerc", "comment": "Leclerc keeps crashing under pressure, Ferrari need someone reliable"},
    {"driver": "Leclerc", "comment": "That pole lap from Charles was one of the greatest I've ever seen"},
    {"driver": "Leclerc", "comment": "I feel genuinely sorry for Leclerc, Ferrari always let him down"},
    {"driver": "Leclerc", "comment": "Charles drove an absolutely perfect race today, flawless"},
    {"driver": "Leclerc", "comment": "Leclerc is so overrated, he only looks good in a fast car"},
    {"driver": "Leclerc", "comment": "The Leclerc and Hamilton Ferrari partnership is going to be epic"},
    {"driver": "Leclerc", "comment": "Charles deserved that championship, Ferrari robbed him"},
    {"driver": "Leclerc", "comment": "Leclerc qualifying pace is unreal, nobody touches him on Saturdays"},
    {"driver": "Leclerc", "comment": "I'm so frustrated watching Leclerc make the same mistakes again"},
    {"driver": "Sainz", "comment": "Carlos is so underrated, consistently delivers when it matters"},
    {"driver": "Sainz", "comment": "Sainz deserved that win more than anyone, what a drive"},
    {"driver": "Piastri", "comment": "Piastri is going to be a world champion, mark my words"},
    {"driver": "Piastri", "comment": "Oscar is so calm under pressure, reminds me of Alonso"},
    {"driver": "Alonso", "comment": "Fernando is still the smartest driver on the grid at 42"},
    {"driver": "Alonso", "comment": "Alonso should have had at least four titles, politics robbed him"},
    {"driver": "Russell", "comment": "Russell is clinical and fast, perfect future champion material"},
    {"driver": "Russell", "comment": "George needs to stop complaining on the radio and just drive"},
    {"driver": "Perez", "comment": "Checo was absolutely nowhere today, Red Bull need to drop him"},
    {"driver": "Perez", "comment": "I feel bad for Perez, the pressure at Red Bull must be immense"},
]

# ── REDDIT FETCH ──
def fetch_reddit_comments():
    try:
        headers = {"User-Agent": "f1sentiment/0.1"}
        response = requests.get(
            "https://www.reddit.com/r/formula1/hot.json",
            headers=headers, timeout=5
        )
        if response.status_code != 200:
            return None, None
        posts = response.json()["data"]["children"]
        post = posts[2]["data"]
        time.sleep(2)
        comments_response = requests.get(
            f"https://www.reddit.com/r/formula1/comments/{post['id']}.json",
            headers=headers, timeout=5
        )
        if comments_response.status_code != 200:
            return None, None
        comments_raw = comments_response.json()[1]["data"]["children"]
        comments = []
        for c in comments_raw[:30]:
            if c["kind"] == "t1":
                body = c["data"]["body"]
                if len(body) > 10 and body.isascii():
                    comments.append({"driver": "Reddit", "comment": body})
        return comments, post["title"]
    except:
        return None, None

# ── LIVE REDDIT BUTTON ──
st.subheader("🔴 Live Reddit Comments")
if st.button("Fetch Live Reddit Comments", key="reddit_btn"):
    with st.spinner("Fetching from Reddit..."):
        reddit_comments, post_title = fetch_reddit_comments()
    if reddit_comments:
        st.success(f"Fetched from: {post_title}")
        comments_to_analyse = reddit_comments
        st.session_state["comments"] = reddit_comments
        st.session_state["source"] = "reddit"
    else:
        st.warning("Reddit unavailable — using default dataset instead.")
        comments_to_analyse = default_comments
        st.session_state["comments"] = default_comments
        st.session_state["source"] = "default"
else:
    comments_to_analyse = st.session_state.get("comments", default_comments)

# ── ANALYSE ──
st.subheader("⏳ Analysing emotions...")
results = []
for item in comments_to_analyse:
    emotion = analyser(item["comment"])[0]
    results.append({
        "Driver": item["driver"],
        "Comment": item["comment"],
        "Emotion": emotion["label"] if emotion["score"] >= 0.7 else "uncertain",
        "Confidence": round(emotion["score"], 2)
    })

df = pd.DataFrame(results)

# ── SECTION 1: Raw Data ──
st.subheader("📊 All Comments & Emotions")
st.dataframe(df, use_container_width=True)

# ── SECTION 2: Emotion Distribution ──
st.subheader("🎭 What Emotions Do Fans Express?")
emotion_counts = df["Emotion"].value_counts().reset_index()
emotion_counts.columns = ["Emotion", "Count"]
fig1 = px.bar(emotion_counts, x="Emotion", y="Count", color="Emotion")
st.plotly_chart(fig1, use_container_width=True)

# ── SECTION 3: Per Driver ──
st.subheader("🏎 Emotion Breakdown Per Driver")
fig2 = px.histogram(df, x="Driver", color="Emotion", barmode="group")
st.plotly_chart(fig2, use_container_width=True)

# ── SECTION 4: Most Loved Driver ──
st.subheader("💚 Most Positively Talked About Driver")
driver_confidence = df[df["Emotion"] != "uncertain"].groupby("Driver")["Confidence"].mean().reset_index()
driver_confidence.columns = ["Driver", "Avg Confidence"]
driver_confidence = driver_confidence.sort_values("Avg Confidence", ascending=False)
fig3 = px.bar(driver_confidence, x="Driver", y="Avg Confidence", color="Driver")
st.plotly_chart(fig3, use_container_width=True)
most_loved = driver_confidence.iloc[0]["Driver"]
st.write(f"📌 Finding: **{most_loved}** generates the most confident emotional reactions from fans.")

# ── SECTION 5: Most Negative Reactions ──
st.subheader("🔴 Which Driver Gets The Most Negative Reactions?")
negative_emotions = ["anger", "sadness", "disgust", "fear"]
df["IsNegative"] = df["Emotion"].isin(negative_emotions)
negative_counts = df.groupby("Driver")["IsNegative"].sum().reset_index()
negative_counts.columns = ["Driver", "Negative Comments"]
negative_counts = negative_counts.sort_values("Negative Comments", ascending=False)
fig4 = px.bar(negative_counts, x="Driver", y="Negative Comments", color="Driver")
st.plotly_chart(fig4, use_container_width=True)
most_negative = negative_counts.iloc[0]["Driver"]
st.write(f"📌 Finding: **{most_negative}** receives the most negative emotional reactions from fans.")