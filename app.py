# app.py
import streamlit as st
from moon_utils import detect_location, get_lat_lon, get_timezone, get_moon_phases
from classifier_utils import fetch_and_classify
from events_utils import get_astronomy_events
from eclipse_utils import get_next_eclipses_from_csv
from constellation_utils import detect_and_match
import chatbot_utils
from astronomy_data import build_astronomical_data

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.cache_data.clear()
st.set_page_config(layout="wide", page_title="Astronomy Dashboard")

# --- 1) LOCATION RESOLUTION ---
loc = st.text_input("Enter location (city)", placeholder="e.g. Tokyo")
if loc:
    try:
        lat, lon, city = get_lat_lon(loc)
        tz = get_timezone(lat, lon)
    except:
        lat, lon, tz, city = detect_location()
else:
    lat, lon, tz, city = detect_location()
st.markdown(f"## 📍 {city}")

# --- 2) LOAD & CACHE ASTRO DATA ---
@st.cache_data(show_spinner=False)
def load_astronomy(city_name):
    return build_astronomical_data(city_name=city_name)

data, embeddings, chunks = load_astronomy(city)
import chatbot_utils

# --- 3) FETCH FORECAST, MOON, ECLIPSES, EVENTS ---
daily_df, hourly_df = fetch_and_classify(lat, lon)
moon_df = get_moon_phases(lat, lon, tz)
eclipses = get_next_eclipses_from_csv()
events = get_astronomy_events()
# Build an “astro context” dict for the bot:
astro_context = {
    "eclipses": {
        "solar": {"date": eclipses["next_solar"]},
        "lunar": {"date": eclipses["next_lunar"]},
    },
    "moon_phases": moon_df.to_dict(orient="records"),
    "events": events,
    # for “tomorrow’s forecast” intent:
    "forecast_daily": daily_df[["Date", "label"]]
                      .rename(columns={"Date":"date", "label_str":"label"})
                      .to_dict(orient="records")
}

# Load embeddings & models into chatbot_utils globals:
chatbot_utils.load_embeddings_and_model(
    embeddings, chunks, astro_context,
    embedder_name="all-MiniLM-L6-v2",      # or your chosen embedder
    llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)


# Ensure Date column is python date
daily_df['Date'] = pd.to_datetime(daily_df['Date']).dt.date
# Map numeric labels to descriptive strings
label_map = {0: 'Bad', 1: 'Fair', 2: 'Good', 3: 'Excellent', 'sun': 'Sun'}
daily_df['label_str'] = daily_df['label'].map(label_map)
hourly_df['label_str'] = hourly_df['label'].map(label_map)

# daily labels 0–3 → '0.png'–'3.png'
daily_df['img_file'] = daily_df['label'].apply(lambda i: f"{i}.png")
daily_df['img_file'] = daily_df['label'].apply(lambda i: f"{i}.png")

# hourly filename mapping
def make_hourly_filename(lbl):
    return f"{lbl}.png"
hourly_df['img_file'] = hourly_df['label'].apply(make_hourly_filename)

# Precompute hours by date to avoid repeated filtering
hourly_df['Date'] = pd.to_datetime(hourly_df['Time'], format='%H:%M:%S').dt.date
hours_by_date = {
    d: g.sort_values('Time').reset_index(drop=True)
    for d, g in hourly_df.groupby('Date')
}

# --- 5) LAYOUT ---
col_main, col_side = st.columns([2,1])

with col_main:
    # 7-Day Forecast
    st.markdown("### 7‑Day Stargazing Forecast")
    if 'sel_day' not in st.session_state:
        st.session_state.sel_day = pd.to_datetime(daily_df['Date'].iloc[0]).date()
        st.session_state.hr_index = 0

    day_cols = st.columns(7)
    for i, row in daily_df.iterrows():
        with day_cols[i]:
            if st.button(pd.to_datetime(row['Date']).strftime('%a'), key=f"day{i}"):
                st.session_state.sel_day = pd.to_datetime(row['Date']).date()
                st.session_state.hr_index = 0  # reset hourly index on new day
            st.image(f"media/{row['img_file']}", width=50)

    detail_rows = daily_df[daily_df['Date'] == st.session_state.sel_day]
    if not detail_rows.empty:
        detail = detail_rows.iloc[0]
        st.markdown(f"### Details for {st.session_state.sel_day.strftime('%A, %B %d')}")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Quality", detail['label_str'])
        m2.metric("Temp (°C)", f"{detail['temperature_2m']:.1f}")
        m3.metric("Cloud (%)", f"{detail['cloud_cover']:.0f}")
        m4.metric("Illumination (%)", f"{detail['Illuminated (%)']:.0f}")
        m5.metric("Sunrise", detail['Day Starts'].strftime('%H:%M'))
        m6.metric("Sunset", detail['Day Ends'].strftime('%H:%M'))
        t1, t2 = st.columns(2)
        twilight_start = (pd.to_datetime(detail['Day Starts'].strftime('%H:%M'), format='%H:%M') - pd.Timedelta(minutes=30)).time()
        twilight_end   = (pd.to_datetime(detail['Day Ends'].strftime('%H:%M'),   format='%H:%M') - pd.Timedelta(minutes=50)).time()
        t1.metric("Twilight Start", twilight_start.strftime('%H:%M'))
        t2.metric("Twilight End",   twilight_end.strftime('%H:%M'))

    # Hourly strip
    st.markdown(f"### Hourly for {st.session_state.sel_day.strftime('%A, %b %d')}")
    hrs = hourly_df[pd.to_datetime(hourly_df['Time']).dt.date == st.session_state.sel_day]
    if 'hr_index' not in st.session_state:
        st.session_state.hr_index = 0
    prev, nxt = st.columns([1,1])
    with prev:
        if st.button("← Prev Hours"):
            st.session_state.hr_index = max(0, st.session_state.hr_index - 6)
    with nxt:
        if st.button("Next Hours →"):
            st.session_state.hr_index = min(len(hrs) - 6, st.session_state.hr_index + 6)
    window = hrs.iloc[st.session_state.hr_index:st.session_state.hr_index+6].reset_index(drop=True)
    win_cols = st.columns(6)
    for idx, hr in window.iterrows():
        with win_cols[idx]:
            st.image(f"media/{hr['img_file']}", width=40)
            st.caption(pd.to_datetime(hr['Time']).strftime('%H:%M'))

    # Chatbot UI (disabled)
    st.markdown("### 💬 Chatbot")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input("Ask a question…", key="chat_input_main")
        submitted = st.form_submit_button("Send")
        if submitted and user_q:
            # query_text returns a dict {type, answer}
            result = chatbot_utils.query_text(user_q)
            answer = result.get("answer", "Sorry, no answer.")
            st.session_state.chat_history.append((user_q, answer))
            st.experimental_rerun()


with col_side:
 # Moon Phases
    st.markdown("### 🌙 Moon Phases")
    mcols = st.columns(2)
    for i, phase in enumerate(['New Moon','Full Moon']):
        with mcols[i]:
            ph = moon_df[moon_df.phase == phase].iloc[0]
            st.image(f"media/{phase.lower().replace(' ','_')}.png", width=60)
            st.write(ph.datetime.strftime('%Y-%m-%d'))

    # Eclipses
    st.markdown("### 🌖 Eclipses")
    ecols = st.columns(2)
    with ecols[0]:
        st.image("media/solar.png", width=50)
        st.write(f"Next Solar: {eclipses['next_solar']}")
    with ecols[1]:
        st.image("media/lunar.png", width=50)
        st.write(f"Next Lunar: {eclipses['next_lunar']}")

    # Events
    st.markdown("### ✨ Upcoming Events")
    for ev in events:
        with st.expander(f"{ev['name']} — {ev['date']}"):
            st.write(ev.get('description','No description available'))

    # Constellation Finder
    st.markdown("### 🔭 Constellation Finder")
    uploaded = st.file_uploader("Upload night‑sky image", type=["png","jpg","jpeg"], key="cn_upload")
    th = st.slider("Threshold", 0, 255, 200, key="cn_thresh")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        orig = np.array(img)
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        coords, _ = detect_and_match(gray, star_thresh=th)
        fig, ax = plt.subplots(); ax.imshow(orig); ax.scatter(coords[:,0],coords[:,1],c='red',s=20); ax.set_axis_off()
        st.pyplot(fig)
        if st.button("Proceed", key="cn_proceed"):
            _, top3 = detect_and_match(gray, star_thresh=th)
            for i, (ratio, score, km, c, out) in enumerate(top3, 1):
                st.write(f"{i}. {c['name']} (ratio={ratio:.2f}, score={score:.3f})")
