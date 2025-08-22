import streamlit as st
import pandas as pd
import sqlite3
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
import requests
import random

st.set_page_config(page_title="🌈 Mental Health Mood Journal", layout="wide")
st.title("🌈 Mental Health Mood Journal")

# -----------------------------
# Mood input
# -----------------------------
mood = st.slider("How's your mood today? (1 = 😞, 10 = 😄)", 1, 10, 5)
notes = st.text_area("Write about your day...")

# -----------------------------
# Database Setup (safe)
# -----------------------------
conn = sqlite3.connect("mood_journal.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS entries (
    entry_date TEXT,
    mood INTEGER,
    notes TEXT
)
""")
conn.commit()

# Add missing columns safely
for col, col_type in [("sentiment","TEXT"), ("polarity","REAL")]:
    try:
        c.execute(f"ALTER TABLE entries ADD COLUMN {col} {col_type}")
    except sqlite3.OperationalError:
        pass
conn.commit()

# -----------------------------
# Motivational Quotes
# -----------------------------
DEFAULT_QUOTES = [
    "Keep going! Every day is a new chance. 🌈",
    "You are stronger than you think. 💪",
    "Small steps count, keep moving forward. 🌱",
    "Breathe. You’ve made it this far. 🌸",
    "Your feelings are valid. 💖"
]

def get_online_quote():
    try:
        response = requests.get("https://api.quotable.io/random", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return f'"{data["content"]}" — {data["author"]}'
    except:
        pass
    return random.choice(DEFAULT_QUOTES)

def get_mood_tip(mood_value, streak_days):
    quote = get_online_quote()
    if mood_value <= 4:
        tip = "💡 Low mood tip: "
    elif mood_value <= 7:
        tip = "💡 Encouragement: "
    else:
        tip = "💡 Positive vibes: "
    if streak_days >= 5:
        tip += f"(🔥 {streak_days}-day streak!) "
    return tip + quote

# -----------------------------
# Analyze Sentiment
# -----------------------------
if st.button("Analyze Sentiment"):
    if notes.strip():
        try:
            blob = TextBlob(notes)
            polarity = blob.sentiment.polarity
            sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        except Exception:
            polarity, sentiment_label = 0, "Neutral"

        if sentiment_label == "Positive":
            st.success(f"Sentiment: {sentiment_label} (Polarity: {polarity:.2f})")
        elif sentiment_label == "Negative":
            st.error(f"Sentiment: {sentiment_label} (Polarity: {polarity:.2f})")
        else:
            st.info(f"Sentiment: {sentiment_label} (Polarity: {polarity:.2f})")

        st.info(get_mood_tip(mood, 0))  # streak=0 by default here
    else:
        st.warning("Please write something before analyzing.")

# -----------------------------
# Save Entry
# -----------------------------
if st.button("Save Entry"):
    if not notes.strip():
        st.warning("Please write some notes before saving.")
    else:
        try:
            blob = TextBlob(notes)
            polarity = blob.sentiment.polarity
            sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        except Exception:
            polarity, sentiment_label = 0, "Neutral"

        c.execute("INSERT INTO entries (entry_date,mood,notes,sentiment,polarity) VALUES (?,?,?,?,?)",
                  (str(date.today()), mood, notes, sentiment_label, polarity))
        conn.commit()
        st.success("Entry saved!")
        st.info(get_mood_tip(mood, 0))

# -----------------------------
# Load History
# -----------------------------
try:
    history = pd.read_sql("SELECT * FROM entries", conn)
except Exception:
    history = pd.DataFrame()

if not history.empty:
    history['entry_date'] = pd.to_datetime(history['entry_date'], errors='coerce')
    history = history.dropna(subset=['entry_date']).sort_values('entry_date', ascending=False)

    # -----------------------------
    # Streak calculation
    # -----------------------------
    streak = 0
    today_dt = pd.to_datetime(date.today())
    for i, row in enumerate(history['entry_date']):
        expected_date = today_dt - pd.Timedelta(days=i)
        if pd.notnull(row) and row.date() == expected_date.date():
            streak += 1
        else:
            break
    st.subheader(f"🔥 Current Streak: {streak} day(s) logged consecutively!")
    if streak >= 5:
        st.success(f"🔥 Amazing! {streak}-day streak! Keep it up!")
    elif streak >= 3:
        st.info(f"✨ Good job! {streak}-day streak going strong!")

    # -----------------------------
    # Calendar Heatmap
    # -----------------------------
    try:
        history['Week'] = history['entry_date'].dt.isocalendar().week
        history['Weekday'] = history['entry_date'].dt.weekday
        month_data = history[history['entry_date'].dt.month == today_dt.month]

        if not month_data.empty:
            month_data = month_data.groupby(['Week','Weekday'], as_index=False)['mood'].mean()
            pivot = month_data.pivot(index='Week', columns='Weekday', values='mood')
            pivot = pivot.reindex(columns=[0,1,2,3,4,5,6])
            fig = px.imshow(
                pivot,
                color_continuous_scale='YlOrRd',
                labels=dict(x="Weekday", y="Week", color="Mood"),
                x=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
                y=pivot.index
            )
            fig.update_xaxes(side="top")
            st.subheader("📅 Mood Calendar View")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No entries for this month yet.")
    except Exception:
        st.write("⚠️ Could not generate calendar view.")

    # -----------------------------
    # Mood History Table
    # -----------------------------
    st.subheader("📅 Mood History")
    display_cols = [col for col in ['entry_date','mood','notes','sentiment','polarity'] if col in history.columns]
    st.dataframe(history[display_cols])

    # -----------------------------
    # Mood Trend Chart (Enhanced)
    # -----------------------------
    st.subheader("📈 Mood Trend (with Rolling Average)")
    try:
        history_sorted = history.sort_values('entry_date')
        history_sorted['mood_roll'] = history_sorted['mood'].rolling(window=3, min_periods=1).mean()
        colors = history_sorted['sentiment'].map({'Positive':'green', 'Neutral':'blue', 'Negative':'red'}).fillna("gray")

        fig3, ax = plt.subplots(figsize=(10,4))
        ax.scatter(history_sorted['entry_date'], history_sorted['mood'], color=colors, s=80)
        ax.plot(history_sorted['entry_date'], history_sorted['mood_roll'], color='orange', linewidth=2, label='3-day rolling avg')
        ax.set_xlabel("Date")
        ax.set_ylabel("Mood (1-10)")
        ax.set_ylim(0,10)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig3)
    except Exception:
        st.write("⚠️ Could not generate mood trend chart.")
else:
    st.write("No entries yet.")

conn.close()
