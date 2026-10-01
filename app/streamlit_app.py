import base64
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Instagram Engagement Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.npz"
META_PATH = ROOT_DIR / "models" / "model_meta.json"
TEAM_IMAGE_DIR = ROOT_DIR / "assets" / "team"

# Team profiles shown in the final About section.
# Photos are stored locally so the deployed app does not depend on external avatar services.
TEAM_MEMBERS = [
    {
        "name": "Youssef Mazher",
        "image": TEAM_IMAGE_DIR / "youssef-mazher.jpg",
        "linkedin": "https://www.linkedin.com/in/youssef-mazher/?isSelfProfile=false",
        "github": "https://github.com/youssef-mazher",
    },
    {
        "name": "Youssef Zizo",
        "image": TEAM_IMAGE_DIR / "youssef-zizo.jpg",
        "linkedin": "https://www.linkedin.com/in/youssef-zizo-80034a359/?isSelfProfile=false",
        "github": "https://github.com/youssefzizo757-yz",
    },
    {
        "name": "Hashim Elhelo",
        "image": TEAM_IMAGE_DIR / "hashim-elhelo.jpg",
        "linkedin": "https://www.linkedin.com/in/hashim-elhelo-034a5b177/?isSelfProfile=false",
        "github": "https://github.com/hashemelhelo2827",
    },
    {
        "name": "Poula Essam",
        "image": TEAM_IMAGE_DIR / "poula-essam.jpg",
        "linkedin": "https://www.linkedin.com/in/poula-essam-257279314/?isSelfProfile=true",
        "github": "https://github.com/PoulaEssam33",
    },
]


def image_to_data_uri(path):
    if isinstance(path, str) and path.startswith(("http://", "https://")):
        return path
    path = Path(path)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@st.cache_resource
def load_model():
    meta = json.loads(open(META_PATH, "r", encoding="utf-8").read())
    model = np.load(MODEL_PATH, allow_pickle=False)
    return model, meta


model, meta = load_model()


def bucket(hour):
    if hour in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    if hour in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    return "normal"


def nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def last_weekday(year, month, weekday):
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def us_holidays(year):
    days = set()

    # Major US federal holidays used by the original project.
    fixed = [(1, 1), (6, 19), (7, 4), (11, 11), (12, 25)]
    for month, day in fixed:
        d = date(year, month, day)
        days.add(d)
        if d.weekday() == 5:
            days.add(d - timedelta(days=1))
        elif d.weekday() == 6:
            days.add(d + timedelta(days=1))

    days.update(
        {
            nth_weekday(year, 1, 0, 3),   # MLK Day
            nth_weekday(year, 2, 0, 3),   # Presidents' Day
            last_weekday(year, 5, 0),     # Memorial Day
            nth_weekday(year, 9, 0, 1),   # Labor Day
            nth_weekday(year, 10, 0, 2),  # Columbus/Indigenous Peoples' Day
            nth_weekday(year, 11, 3, 4),  # Thanksgiving
        }
    )
    return days


def make_features(
    followers,
    post_images,
    video,
    carousel,
    length_caption,
    number_hashtags,
    publication_weekday,
    is_weekend,
    is_holiday,
    user_median_engagement,
    user_post_count,
    reach_time_bucket,
    caption_length_bucket,
    hashtag_bucket,
):
    row = {
        "followers": float(followers),
        "post_images": float(post_images),
        "video": float(video),
        "carousel": float(carousel),
        "length_caption": float(length_caption),
        "number_hashtags": float(number_hashtags),
        "is_weekend": float(is_weekend),
        "is_holiday": float(is_holiday),
        "user_median_engagement": float(user_median_engagement),
        "user_post_count": float(user_post_count),
    }

    values = {
        "publication_weekday": str(publication_weekday),
        "reach_time_bucket": str(reach_time_bucket),
        "caption_length_bucket": str(caption_length_bucket),
        "hashtag_bucket": str(hashtag_bucket),
    }

    for col, categories in meta["categorical_values"].items():
        for category in categories:
            row[f"{col}__{category}"] = float(values[col] == category)

    return np.array(
        [row[name] for name in meta["feature_names"]],
        dtype=np.float32,
    )


def predict_one(x):
    total = 0.0
    n_trees = int(meta["n_trees"])

    for i in range(n_trees):
        left = model[f"cl_{i}"]
        right = model[f"cr_{i}"]
        feature = model[f"feat_{i}"]
        threshold = model[f"thr_{i}"]
        value = model[f"val_{i}"]

        node = 0
        while left[node] != -1:
            f = int(feature[node])
            if x[f] <= threshold[node]:
                node = int(left[node])
            else:
                node = int(right[node])

        total += float(value[node])

    return max(0.0, total / n_trees)

# -----------------------------
# UI — Editorial Performance Lab
# -----------------------------
# A full website-style presentation built with native Streamlit + CSS.
# The prediction logic above is intentionally left unchanged.

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg: #0B0D0F;
            --bg-2: #121518;
            --paper: #F4F0E7;
            --paper-2: #EDE7DA;
            --ink: #111315;
            --muted: #A9A8A1;
            --muted-dark: #666760;
            --line: rgba(244, 240, 231, 0.16);
            --line-dark: rgba(17, 19, 21, 0.16);
            --orange: #FF5A36;
            --orange-dark: #D94222;
            --mint: #B9FBCB;
            --blue: #9CB6FF;
        }

        html { scroll-behavior: smooth; }

        .stApp {
            background: var(--bg);
            color: var(--paper);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 0.8rem;
            padding-bottom: 0;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer, [data-testid="stToolbar"] {
            visibility: hidden;
        }

        h1, h2, h3, h4, h5, p, div, span, label, button, input, textarea {
            font-family: "Space Grotesk", sans-serif;
        }

        code, .mono, .eyebrow, .section-index, .metric-kicker,
        .feature-tag, .micro-label, .step-index, .footer-meta {
            font-family: "IBM Plex Mono", monospace !important;
        }

        a { text-decoration: none; }

        /* ---------- Navigation ---------- */
        .site-nav {
            min-height: 84px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            margin: .9rem 0 0;
            padding: .78rem .9rem .78rem 1rem;
            background: var(--paper);
            border: 1px solid rgba(244,240,231,.72);
            border-radius: 15px;
            box-shadow: 0 16px 40px rgba(0,0,0,.20);
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: .9rem;
            color: var(--ink);
            min-width: 0;
        }

        .brand-symbol {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            position: relative;
            font-family: "IBM Plex Mono", monospace;
            font-size: .82rem;
            font-weight: 700;
            color: var(--paper);
            background: #111315;
            border: 1px solid #111315;
            box-shadow: 4px 4px 0 var(--orange);
            transform: none;
        }

        .brand-symbol:after {
            content: "";
            position: absolute;
            width: 7px;
            height: 7px;
            right: 5px;
            top: 5px;
            border-radius: 50%;
            background: var(--orange);
        }

        .brand-name {
            color: #111315;
            font-size: 1.16rem;
            font-weight: 700;
            letter-spacing: -.035em;
            line-height: 1.08;
        }

        .brand-sub {
            color: #73736D;
            font-size: .88rem;
            font-weight: 500;
            margin-top: .28rem;
            line-height: 1.2;
        }

        .nav-links {
            display: flex;
            align-items: center;
            gap: .12rem;
        }

        nav.site-nav a.nav-link,
        nav.site-nav a.nav-link:visited,
        nav.site-nav a.nav-link:active {
            color: #252729 !important;
            text-decoration: none !important;
            font-size: .93rem;
            font-weight: 650;
            letter-spacing: -.018em;
            line-height: 1;
            padding: .78rem .82rem;
            border-radius: 8px;
            transition: color .15s ease, background .15s ease, transform .15s ease;
        }

        nav.site-nav a.nav-link:hover {
            color: #111315 !important;
            text-decoration: none !important;
            background: rgba(17,19,21,.075);
            transform: translateY(-1px);
        }

        .nav-status {
            min-height: 43px;
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            color: var(--paper);
            background: #111315;
            border-radius: 9px;
            padding: 0 1rem;
            margin-left: .45rem;
            font-family: "IBM Plex Mono", monospace;
            font-weight: 600;
            font-size: .79rem;
            letter-spacing: .025em;
            white-space: nowrap;
            box-shadow: none;
        }

        .nav-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--orange);
            box-shadow: 0 0 0 3px rgba(255,90,54,.13);
        }

        /* ---------- Hero ---------- */
        .hero-shell {
            min-height: 560px;
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(320px, .85fr);
            gap: 4rem;
            align-items: start;
            padding: 4.8rem 0 4.6rem;
            position: relative;
        }

        .hero-shell > div:first-child {
            padding-top: 1.2rem;
        }

        .hero-shell:before {
            content: "";
            position: absolute;
            width: 330px;
            height: 330px;
            right: 2%;
            top: 13%;
            border: 1px solid rgba(255,90,54,.35);
            border-radius: 50%;
            pointer-events: none;
        }

        .hero-shell:after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            right: 8%;
            top: 26%;
            border-radius: 50%;
            background: rgba(255,90,54,.09);
            filter: blur(1px);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .7rem;
            color: var(--orange);
            text-transform: uppercase;
            letter-spacing: .085em;
            font-size: .78rem;
            font-weight: 600;
            margin-bottom: 1.55rem;
        }

        .eyebrow:before {
            content: "";
            width: 34px;
            height: 2px;
            background: var(--orange);
        }

        .hero-title {
            max-width: 820px;
            font-size: clamp(4rem, 6.6vw, 6.8rem);
            line-height: .91;
            letter-spacing: -.06em;
            font-weight: 700;
            color: var(--paper);
            margin: 0;
        }

        .hero-title .outline {
            color: transparent;
            -webkit-text-stroke: 1px #F4F0E7;
        }

        .hero-title .accent { color: var(--orange); }

        .hero-copy {
            max-width: 720px;
            color: #B8B7B0;
            font-size: 1.12rem;
            line-height: 1.68;
            margin-top: 2.05rem;
        }

        .hero-actions {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: .9rem;
            margin-top: 2.2rem;
        }

        .cta-primary, .cta-secondary {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 54px;
            padding: 0 1.25rem;
            font-size: .9rem;
            font-weight: 700;
            transition: transform .16s ease, background .16s ease, color .16s ease;
        }

        .cta-primary {
            background: var(--orange);
            color: white !important;
            border: 1px solid var(--orange);
        }

        .cta-primary:hover {
            background: #FF6F50;
            transform: translateY(-2px);
        }

        .cta-secondary {
            color: var(--paper) !important;
            border: 1px solid rgba(244,240,231,.36);
        }

        .cta-secondary:hover {
            border-color: var(--paper);
            transform: translateY(-2px);
        }

        .hero-rail {
            position: relative;
            z-index: 2;
            border-left: 1px solid var(--line);
            padding-left: 2rem;
        }

        .signal-card {
            background: #111417;
            border: 1px solid var(--line);
            padding: 1.4rem;
            margin-bottom: 1rem;
        }

        .signal-card.compact {
            padding: 1.05rem 1.25rem 1.15rem;
            margin-bottom: 0;
        }

        .signal-card.compact .signal-grid {
            margin-top: .6rem;
        }

        .signal-card.compact .signal-mini {
            padding-top: .6rem;
        }

        .signal-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.35rem;
        }

        .metric-kicker {
            color: #A8A79F;
            text-transform: uppercase;
            letter-spacing: .075em;
            font-size: .72rem;
        }

        .live-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            background: var(--mint);
            border-radius: 50%;
            box-shadow: 0 0 0 5px rgba(185,251,203,.08);
        }

        .signal-big {
            font-size: 2.55rem;
            line-height: 1;
            font-weight: 700;
            letter-spacing: -.05em;
            margin-bottom: .45rem;
        }

        .signal-caption {
            color: #A7A59E;
            font-size: .88rem;
            line-height: 1.5;
        }

        .signal-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: .6rem;
            margin-top: 1.15rem;
        }

        .signal-mini {
            border-top: 1px solid var(--line);
            padding-top: .7rem;
        }

        .signal-mini span {
            display: block;
            color: #8D8C86;
            font-size: .72rem;
            margin-bottom: .3rem;
        }

        .signal-mini strong {
            color: var(--paper);
            font-size: 1.08rem;
        }

        .signal-bars {
            height: 72px;
            display: flex;
            align-items: flex-end;
            gap: 5px;
            margin-top: .9rem;
        }

        .signal-bars i {
            flex: 1;
            min-width: 6px;
            background: #2B2F33;
            display: block;
        }

        .signal-bars i:nth-child(2n) { background: #3A3F44; }
        .signal-bars i:nth-child(5), .signal-bars i:nth-child(8) { background: var(--orange); }

        /* ---------- Light section wrappers ---------- */
        .section-anchor { scroll-margin-top: 28px; }

        .light-band {
            background: var(--paper);
            color: var(--ink);
            margin-left: calc(50% - 50vw);
            margin-right: calc(50% - 50vw);
            padding: 5rem max(calc((100vw - 1240px)/2), 1rem);
        }

        .dark-band {
            background: var(--bg);
            color: var(--paper);
            margin-left: calc(50% - 50vw);
            margin-right: calc(50% - 50vw);
            padding: 5rem max(calc((100vw - 1240px)/2), 1rem);
        }

        .section-heading {
            display: grid;
            grid-template-columns: 120px 1fr;
            gap: 2rem;
            align-items: start;
            margin-bottom: 2.4rem;
        }

        .section-index {
            color: var(--orange);
            font-size: .82rem;
            letter-spacing: .075em;
            padding-top: .48rem;
            font-weight: 600;
        }

        .section-title {
            margin: 0;
            font-size: clamp(2.2rem, 4.5vw, 4.8rem);
            line-height: .96;
            letter-spacing: -.055em;
            font-weight: 700;
        }

        .section-copy {
            max-width: 710px;
            color: var(--muted-dark);
            line-height: 1.7;
            font-size: 1rem;
            margin-top: 1.1rem;
        }

        .dark-band .section-copy { color: #A4A39D; }

        /* ---------- About ---------- */
        .about-section {
            position: relative;
            overflow: hidden;
        }

        .about-head {
            display: grid;
            grid-template-columns: 150px minmax(0, 1fr);
            gap: 2.2rem;
            align-items: start;
            padding-bottom: 2.8rem;
            border-bottom: 1px solid var(--line-dark);
        }

        .about-index {
            font-family: "IBM Plex Mono", monospace;
            color: var(--orange-dark);
            font-size: .82rem;
            font-weight: 600;
            letter-spacing: .08em;
            text-transform: uppercase;
            padding-top: .5rem;
        }

        .about-head-main {
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(280px, .6fr);
            gap: 3.5rem;
            align-items: end;
        }

        .about-title {
            margin: 0;
            max-width: 820px;
            color: var(--ink);
            font-size: clamp(2.8rem, 5vw, 5.2rem);
            line-height: .95;
            letter-spacing: -.06em;
            font-weight: 700;
        }

        .about-intro {
            margin: 0 0 .35rem;
            color: #666760;
            font-size: 1rem;
            line-height: 1.72;
            max-width: 390px;
        }

        .about-thesis {
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(320px, .95fr);
            gap: 4rem;
            align-items: stretch;
            margin-top: 2.7rem;
        }

        .about-quote {
            min-height: 250px;
            background: var(--ink);
            color: var(--paper);
            padding: 2rem 2.15rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
        }

        .about-quote:after {
            content: "";
            position: absolute;
            width: 190px;
            height: 190px;
            right: -70px;
            bottom: -95px;
            border: 1px solid rgba(255,90,54,.55);
            border-radius: 50%;
        }

        .about-quote-kicker {
            font-family: "IBM Plex Mono", monospace;
            color: var(--orange);
            font-size: .78rem;
            letter-spacing: .08em;
            text-transform: uppercase;
            font-weight: 600;
        }

        .about-quote p {
            margin: 0;
            max-width: 670px;
            color: var(--paper);
            font-size: clamp(1.65rem, 2.6vw, 2.65rem) !important;
            line-height: 1.08 !important;
            letter-spacing: -.045em;
            font-weight: 600;
        }

        .about-quote p em {
            color: var(--orange);
            font-style: normal;
        }

        .about-summary {
            border-top: 1px solid var(--line-dark);
            display: grid;
            grid-template-rows: repeat(3, 1fr);
        }

        .about-summary-item {
            display: grid;
            grid-template-columns: 44px 1fr;
            gap: 1rem;
            align-content: center;
            padding: 1.15rem 0;
            border-bottom: 1px solid var(--line-dark);
        }

        .about-summary-no {
            font-family: "IBM Plex Mono", monospace;
            color: var(--orange-dark);
            font-size: .76rem;
            font-weight: 600;
            padding-top: .12rem;
        }

        .about-summary-item strong {
            display: block;
            color: var(--ink);
            font-size: 1.02rem;
            letter-spacing: -.018em;
            margin-bottom: .32rem;
        }

        .about-summary-item span {
            display: block;
            color: #6D6E68;
            font-size: .88rem;
            line-height: 1.55;
        }

        .about-pipeline-head {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 2rem;
            margin-top: 3.2rem;
            margin-bottom: 1.15rem;
        }

        .about-pipeline-head strong {
            color: var(--ink);
            font-size: 1.15rem;
            letter-spacing: -.025em;
        }

        .about-pipeline-head span {
            font-family: "IBM Plex Mono", monospace;
            color: #777871;
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .07em;
        }

        .architecture-strip {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            border: 1px solid var(--line-dark);
            background: rgba(255,255,255,.18);
        }

        .architecture-node {
            min-height: 138px;
            padding: 1.2rem 1.15rem 1.15rem;
            border-right: 1px solid var(--line-dark);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
        }

        .architecture-node:last-child { border-right: 0; }

        .architecture-node:before {
            content: "";
            position: absolute;
            top: -1px;
            left: -1px;
            width: 42%;
            height: 3px;
            background: var(--orange);
            opacity: .9;
        }

        .architecture-node:not(:last-child):after {
            content: "→";
            position: absolute;
            right: -13px;
            top: 50%;
            transform: translateY(-50%);
            width: 26px;
            height: 26px;
            display: grid;
            place-items: center;
            background: var(--paper);
            color: var(--orange-dark);
            border: 1px solid var(--line-dark);
            border-radius: 50%;
            font-family: "IBM Plex Mono", monospace;
            font-size: .74rem;
            z-index: 2;
        }

        .micro-label {
            color: #777871;
            font-size: .68rem;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .architecture-node strong {
            max-width: 190px;
            color: var(--ink);
            font-size: 1.02rem;
            line-height: 1.25;
            letter-spacing: -.025em;
        }

        /* ---------- Predictor ---------- */
        .predictor-intro {
            margin-bottom: 1.35rem;
        }

        .predictor-kicker {
            display: inline-block;
            color: var(--orange);
            font-family: "IBM Plex Mono", monospace;
            text-transform: uppercase;
            font-size: .67rem;
            letter-spacing: .08em;
            margin-bottom: .55rem;
        }

        .predictor-title {
            margin: 0;
            font-size: clamp(2.4rem, 4.4vw, 4.2rem);
            line-height: .98;
            letter-spacing: -.055em;
        }

        .predictor-sub {
            color: #A4A39D;
            max-width: 680px;
            line-height: 1.65;
            margin-top: .8rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #111417 !important;
            border: 1px solid rgba(244,240,231,.14) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] h3,
        div[data-testid="stVerticalBlockBorderWrapper"] h4 {
            color: var(--paper);
            letter-spacing: -.03em;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] p,
        div[data-testid="stVerticalBlockBorderWrapper"] small {
            color: #8F8E89;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input,
        div[data-baseweb="select"] > div {
            background: #0D1012 !important;
            color: var(--paper) !important;
            border: 1px solid rgba(244,240,231,.16) !important;
            border-radius: 0 !important;
        }

        div[data-testid="stNumberInput"] button {
            background: #15191C !important;
            color: var(--paper) !important;
            border-radius: 0 !important;
        }

        label[data-testid="stWidgetLabel"] p {
            color: #D7D3CA !important;
            font-size: .82rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stSlider"] [role="slider"] {
            background-color: var(--orange) !important;
        }

        .auto-strip {
            display: flex;
            align-items: center;
            gap: .5rem;
            flex-wrap: wrap;
            padding: .9rem 1rem;
            border: 1px solid var(--line);
            background: #101316;
            margin: .9rem 0 1rem;
        }

        .auto-label {
            color: var(--orange);
            font-family: "IBM Plex Mono", monospace;
            font-size: .64rem;
            text-transform: uppercase;
            letter-spacing: .07em;
            margin-right: .15rem;
        }

        .feature-tag {
            color: #C5C2BA;
            border: 1px solid rgba(244,240,231,.14);
            padding: .34rem .48rem;
            font-size: .64rem;
        }

        .stButton > button {
            min-height: 54px;
            border-radius: 0 !important;
            border: 1px solid var(--orange) !important;
            background: var(--orange) !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: .92rem !important;
            box-shadow: none !important;
            transition: transform .14s ease, background .14s ease !important;
        }

        .stButton > button:hover {
            background: #FF6A48 !important;
            transform: translateY(-2px);
        }

        /* ---------- Results ---------- */
        .result-wrap {
            border: 1px solid rgba(244,240,231,.18);
            background: #0F1214;
            min-height: 320px;
            padding: 1.4rem;
            position: relative;
            overflow: hidden;
        }

        .result-wrap:after {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            border-radius: 50%;
            background: var(--orange);
            opacity: .09;
            right: -90px;
            top: -80px;
        }

        .result-kicker {
            color: var(--orange);
            font-family: "IBM Plex Mono", monospace;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-size: .65rem;
        }

        .result-value {
            color: var(--paper);
            font-size: clamp(4.6rem, 8vw, 7.5rem);
            line-height: .95;
            letter-spacing: -.07em;
            font-weight: 700;
            margin: 1.5rem 0 .5rem;
        }

        .result-copy {
            color: #9B9A94;
            max-width: 500px;
            font-size: .88rem;
            line-height: 1.55;
        }

        .result-delta {
            display: inline-flex;
            margin-top: 1.1rem;
            padding: .45rem .58rem;
            border: 1px solid rgba(244,240,231,.16);
            color: #D7D3CA;
            font-family: "IBM Plex Mono", monospace;
            font-size: .65rem;
        }

        .result-delta.positive { color: var(--mint); border-color: rgba(185,251,203,.35); }
        .result-delta.negative { color: #FF9D88; border-color: rgba(255,157,136,.35); }

        .benchmark-box {
            border: 1px solid rgba(244,240,231,.16);
            background: #111417;
            padding: 1.2rem;
            min-height: 320px;
        }

        .benchmark-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
        }

        .compare-row { margin-bottom: 1.2rem; }
        .compare-top {
            display: flex;
            justify-content: space-between;
            color: #B7B5AE;
            font-size: .76rem;
            margin-bottom: .45rem;
        }
        .compare-top strong { color: var(--paper); }
        .compare-track {
            height: 8px;
            background: #252A2E;
            overflow: hidden;
        }
        .compare-fill {
            height: 100%;
            background: var(--orange);
        }
        .compare-fill.median { background: #777C82; }

        .result-note {
            color: #85847F;
            font-size: .72rem;
            line-height: 1.5;
            border-top: 1px solid var(--line);
            padding-top: .9rem;
            margin-top: 1rem;
        }

        /* ---------- Method ---------- */
        .method-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0;
            border: 1px solid var(--line-dark);
        }

        .method-card {
            min-height: 245px;
            padding: 1.2rem;
            border-right: 1px solid var(--line-dark);
            position: relative;
        }
        .method-card:last-child { border-right: 0; }

        .step-index {
            color: var(--orange-dark);
            font-size: .68rem;
            letter-spacing: .06em;
        }

        .method-card h4 {
            margin: 4.5rem 0 .5rem;
            font-size: 1.1rem;
            letter-spacing: -.025em;
        }

        .method-card p {
            color: var(--muted-dark);
            font-size: .84rem;
            line-height: 1.55;
            margin: 0;
        }

        .method-card .arrow {
            position: absolute;
            top: 1.1rem;
            right: 1rem;
            color: #AAA79E;
            font-size: 1.3rem;
        }

        .feature-matrix {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            border-top: 1px solid var(--line-dark);
            border-left: 1px solid var(--line-dark);
            margin-top: 2rem;
        }

        .feature-cell {
            padding: 1.1rem;
            border-right: 1px solid var(--line-dark);
            border-bottom: 1px solid var(--line-dark);
        }

        .feature-cell h5 {
            margin: 0 0 .55rem;
            font-size: .9rem;
        }

        .feature-cell p {
            margin: 0;
            color: var(--muted-dark);
            font-size: .78rem;
            line-height: 1.6;
        }

        /* ---------- Model ---------- */
        .model-panel {
            display: grid;
            grid-template-columns: 1.1fr .9fr;
            gap: 4rem;
            align-items: start;
        }

        .model-copy-large {
            font-size: clamp(1.7rem, 3vw, 2.8rem);
            line-height: 1.08;
            letter-spacing: -.045em;
            margin: 0 0 1.2rem;
        }

        .model-body {
            color: #9A9993;
            line-height: 1.7;
            max-width: 620px;
        }

        .metric-stack {
            border-top: 1px solid var(--line);
        }

        .metric-line {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 1rem 0;
            border-bottom: 1px solid var(--line);
        }

        .metric-line span {
            color: #8E8D87;
            font-size: .78rem;
        }

        .metric-line strong {
            font-size: 1.4rem;
            letter-spacing: -.035em;
        }

        .disclaimer {
            margin-top: 2rem;
            border-left: 3px solid var(--orange);
            padding: .8rem 1rem;
            background: rgba(255,90,54,.06);
            color: #AAA9A3;
            font-size: .78rem;
            line-height: 1.55;
        }

        /* ---------- Final About / Team ---------- */
        .team-band {
            background: #0F1214;
            color: var(--paper);
            margin-left: calc(50% - 50vw);
            margin-right: calc(50% - 50vw);
            padding: 5.4rem max(calc((100vw - 1240px)/2), 1rem);
            border-top: 1px solid var(--line);
        }

        .team-head {
            display: grid;
            grid-template-columns: 150px minmax(0, 1fr);
            gap: 2.2rem;
            align-items: start;
            margin-bottom: 2.6rem;
        }

        .team-index {
            font-family: "IBM Plex Mono", monospace;
            color: var(--orange);
            font-size: .82rem;
            font-weight: 600;
            letter-spacing: .08em;
            text-transform: uppercase;
            padding-top: .5rem;
        }

        .team-title {
            margin: 0;
            max-width: 780px;
            color: var(--paper);
            font-size: clamp(2.7rem, 5vw, 5rem);
            line-height: .95;
            letter-spacing: -.06em;
            font-weight: 700;
        }

        .team-intro {
            max-width: 700px;
            color: #9E9D96;
            font-size: 1rem;
            line-height: 1.7;
            margin-top: 1rem;
        }

        .team-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
        }

        .team-card {
            border: 1px solid var(--line);
            background: #121518;
            overflow: hidden;
        }

        .team-card > img.team-photo {
            display: block !important;
            width: 100% !important;
            max-width: none !important;
            height: 300px !important;
            aspect-ratio: auto !important;
            object-fit: cover !important;
            object-position: center 35% !important;
            margin: 0 !important;
            padding: 0 !important;
            background: #181C20;
            filter: saturate(.88) contrast(1.03);
        }

        .team-card-body { padding: 1.15rem; }

        .team-name {
            color: var(--paper);
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: -.025em;
        }

        .team-initials {
            width: 100%;
            height: 300px;
            aspect-ratio: auto;
            display: grid;
            place-items: center;
            background: #181C20;
            color: var(--paper);
            font-family: "IBM Plex Mono", monospace;
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: .08em;
            border-bottom: 1px solid var(--line);
        }

        .team-links {
            display: flex;
            gap: .55rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .team-link {
            display: inline-flex;
            align-items: center;
            min-height: 36px;
            padding: 0 .7rem;
            border: 1px solid rgba(244,240,231,.18);
            color: #D7D4CB !important;
            font-family: "IBM Plex Mono", monospace;
            font-size: .68rem;
            letter-spacing: .035em;
            text-transform: uppercase;
        }

        .team-link:hover {
            border-color: var(--orange);
            color: var(--orange) !important;
        }

        .team-empty {
            border: 1px dashed rgba(244,240,231,.22);
            padding: 1.2rem 1.3rem;
            color: #8E8D87;
            font-size: .88rem;
            line-height: 1.6;
        }

        @media (max-width: 900px) {
            .team-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 700px) {
            .team-head { grid-template-columns: 1fr; gap: .8rem; }
            .team-grid { grid-template-columns: 1fr; }
        }

        /* ---------- Footer ---------- */
        .site-footer {
            border-top: 1px solid var(--line);
            margin-top: 0;
            padding: 2.2rem 0 2.8rem;
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 2rem;
        }

        .footer-title {
            color: var(--paper);
            font-size: 1.1rem;
            font-weight: 700;
        }

        .footer-copy {
            color: #797973;
            font-size: .76rem;
            margin-top: .35rem;
            max-width: 560px;
            line-height: 1.5;
        }

        .footer-meta {
            color: #6F6F69;
            font-size: .62rem;
            text-align: right;
            letter-spacing: .05em;
            text-transform: uppercase;
        }

        .avatar-credit {
            display: inline-block;
            color: #8F8E89 !important;
            font-size: .68rem;
            margin-top: .42rem;
            text-decoration: underline;
            text-underline-offset: 3px;
        }

        .avatar-credit:hover { color: var(--orange) !important; }

        /* ---------- Streamlit expander / metrics ---------- */
        div[data-testid="stExpander"] {
            border: 1px solid rgba(244,240,231,.16) !important;
            border-radius: 0 !important;
            background: #111417 !important;
            overflow: hidden;
        }

        /* Streamlit styles the expander header separately from its body. */
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] details[open] {
            background: #111417 !important;
            color: var(--paper) !important;
        }

        div[data-testid="stExpander"] summary {
            min-height: 54px !important;
            padding: .9rem 1rem !important;
            background: #181C1F !important;
            color: var(--paper) !important;
            border-bottom: 1px solid rgba(244,240,231,.12) !important;
            transition: background .15s ease, border-color .15s ease;
        }

        div[data-testid="stExpander"] summary:hover {
            background: #1D2226 !important;
            border-bottom-color: rgba(255,90,54,.55) !important;
        }

        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary div {
            color: #F4F0E7 !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }

        div[data-testid="stExpander"] summary svg,
        div[data-testid="stExpander"] summary svg path {
            color: var(--orange) !important;
            fill: var(--orange) !important;
            stroke: var(--orange) !important;
            opacity: 1 !important;
        }

        div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background: #111417 !important;
            color: var(--paper) !important;
            padding-top: .8rem !important;
        }

        div[data-testid="stMetric"] {
            background: transparent !important;
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] span {
            color: #B8B7B0 !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div,
        div[data-testid="stMetricValue"] span {
            color: #F4F0E7 !important;
            opacity: 1 !important;
        }

        div[data-testid="stExpander"] [data-testid="stCaptionContainer"],
        div[data-testid="stExpander"] [data-testid="stCaptionContainer"] p,
        div[data-testid="stExpander"] small {
            color: #AAA9A3 !important;
            opacity: 1 !important;
        }

        @media (max-width: 980px) {
            .hero-shell { grid-template-columns: 1fr; gap: 2.5rem; min-height: auto; }
            .hero-rail { border-left: 0; padding-left: 0; max-width: 520px; }
            .hero-shell:before, .hero-shell:after { display: none; }
            .about-head-main, .about-thesis, .model-panel { grid-template-columns: 1fr; gap: 2.2rem; }
            .architecture-strip, .method-grid { grid-template-columns: repeat(2, 1fr); }
            .architecture-node:nth-child(2) { border-right: 0; }
            .architecture-node:nth-child(-n+2) { border-bottom: 1px solid var(--line-dark); }
            .architecture-node:nth-child(2):after { display: none; }
            .method-card:nth-child(2) { border-right: 0; }
            .method-card:nth-child(-n+2) { border-bottom: 1px solid var(--line-dark); }
            .feature-matrix { grid-template-columns: 1fr; }
        }

        @media (max-width: 700px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .nav-links { display: none; }
            .hero-shell { padding: 4rem 0 3.5rem; }
            .hero-title { font-size: clamp(3.4rem, 16vw, 5.3rem); }
            .section-heading { grid-template-columns: 1fr; gap: .5rem; }
            .about-head { grid-template-columns: 1fr; gap: .8rem; }
            .about-title { font-size: clamp(2.6rem, 12vw, 4.2rem); }
            .about-pipeline-head { align-items: flex-start; flex-direction: column; gap: .35rem; }
            .architecture-strip, .method-grid { grid-template-columns: 1fr; }
            .architecture-node { border-right: 0; border-bottom: 1px solid var(--line-dark); min-height: 112px; }
            .architecture-node:last-child { border-bottom: 0; }
            .architecture-node:after { display: none !important; }
            .architecture-node, .method-card { border-right: 0; border-bottom: 1px solid var(--line-dark); }
            .architecture-node:last-child, .method-card:last-child { border-bottom: 0; }
            .feature-matrix { grid-template-columns: 1fr; }
            .site-footer { flex-direction: column; align-items: flex-start; }
            .footer-meta { text-align: left; }
        }

        /* ---------- Final contrast overrides: predictor controls + technical metrics ---------- */
        /* Technical metric labels: force a readable muted light tone on the dark panel. */
        div[data-testid="stExpander"] div[data-testid="stMetric"] label,
        div[data-testid="stExpander"] div[data-testid="stMetric"] label *,
        div[data-testid="stExpander"] div[data-testid="stMetricLabel"],
        div[data-testid="stExpander"] div[data-testid="stMetricLabel"] *,
        div[data-testid="stExpander"] [data-testid="stMetricLabel"],
        div[data-testid="stExpander"] [data-testid="stMetricLabel"] * {
            color: #B8B7B0 !important;
            -webkit-text-fill-color: #B8B7B0 !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }

        /* Technical metric values remain the highest-contrast element. */
        div[data-testid="stExpander"] div[data-testid="stMetricValue"],
        div[data-testid="stExpander"] div[data-testid="stMetricValue"] *,
        div[data-testid="stExpander"] [data-testid="stMetricValue"],
        div[data-testid="stExpander"] [data-testid="stMetricValue"] * {
            color: #F4F0E7 !important;
            -webkit-text-fill-color: #F4F0E7 !important;
            opacity: 1 !important;
        }

        /* Predictor input text and placeholder text. */
        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
            color: #F4F0E7 !important;
            -webkit-text-fill-color: #F4F0E7 !important;
        }

        div[data-testid="stNumberInput"] input::placeholder,
        div[data-testid="stDateInput"] input::placeholder,
        div[data-baseweb="select"] input::placeholder {
            color: #8F918D !important;
            -webkit-text-fill-color: #8F918D !important;
            opacity: 1 !important;
        }

        /* Number-input +/- controls. */
        div[data-testid="stNumberInput"] button,
        div[data-testid="stNumberInput"] button * {
            color: #D9D6CE !important;
            -webkit-text-fill-color: #D9D6CE !important;
        }

        div[data-testid="stNumberInput"] button svg {
            color: #D9D6CE !important;
            opacity: 1 !important;
        }

        div[data-testid="stNumberInput"] button svg path,
        div[data-testid="stNumberInput"] button svg line,
        div[data-testid="stNumberInput"] button svg polyline {
            stroke: #D9D6CE !important;
        }

        /* Select chevron / dropdown indicator. */
        div[data-baseweb="select"] svg {
            color: #D9D6CE !important;
            fill: #D9D6CE !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] svg path {
            fill: #D9D6CE !important;
            stroke: #D9D6CE !important;
        }

        /* Date/calendar controls, including Chromium's native date icon. */
        div[data-testid="stDateInput"] button,
        div[data-testid="stDateInput"] button *,
        div[data-testid="stDateInput"] svg {
            color: #D9D6CE !important;
            -webkit-text-fill-color: #D9D6CE !important;
            opacity: 1 !important;
        }

        div[data-testid="stDateInput"] svg path,
        div[data-testid="stDateInput"] svg line,
        div[data-testid="stDateInput"] svg polyline,
        div[data-testid="stDateInput"] svg rect {
            stroke: #D9D6CE !important;
        }

        div[data-testid="stDateInput"] input::-webkit-calendar-picker-indicator {
            filter: invert(92%) sepia(8%) saturate(159%) hue-rotate(357deg) brightness(105%) contrast(90%);
            opacity: .95 !important;
            cursor: pointer;
        }

        /* Help / tooltip icons used beside predictor labels. */
        [data-testid="stTooltipIcon"],
        [data-testid="stTooltipIcon"] *,
        [data-testid="stWidgetLabel"] button,
        [data-testid="stWidgetLabel"] button * {
            color: #AAA9A3 !important;
            -webkit-text-fill-color: #AAA9A3 !important;
            opacity: 1 !important;
        }

        [data-testid="stTooltipIcon"] svg,
        [data-testid="stWidgetLabel"] button svg {
            color: #AAA9A3 !important;
            fill: #AAA9A3 !important;
            opacity: 1 !important;
        }

        /* Slider value/ticks need to stay readable on the dark predictor panel. */
        div[data-testid="stSlider"] p,
        div[data-testid="stSlider"] span,
        div[data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: #C8C5BD !important;
            -webkit-text-fill-color: #C8C5BD !important;
            opacity: 1 !important;
        }

        div[data-testid="stSlider"] [role="slider"] {
            border-color: var(--orange) !important;
            box-shadow: 0 0 0 3px rgba(255,90,54,.12) !important;
        }

        /* Captions under media-specific predictor controls. */
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"],
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] * {
            color: #9F9E98 !important;
            -webkit-text-fill-color: #9F9E98 !important;
            opacity: 1 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation
st.markdown(
    """
    <nav class="site-nav">
        <div class="brand-lockup">
            <div class="brand-symbol">IG</div>
            <div>
                <div class="brand-name">Engagement Predictor</div>
                <div class="brand-sub">Machine-learning planning tool</div>
            </div>
        </div>
        <div class="nav-links">
            <a class="nav-link" href="#overview">Overview</a>
            <a class="nav-link" href="#predictor">Predict</a>
            <a class="nav-link" href="#method">Method</a>
            <a class="nav-link" href="#model">Model</a>
            <a class="nav-link" href="#about">About</a>
            <span class="nav-status"><span class="nav-status-dot"></span>LIVE DEMO</span>
        </div>
    </nav>
    """,
    unsafe_allow_html=True,
)

# Hero
st.markdown(
    f"""
    <section class="hero-shell">
        <div>
            <div class="eyebrow">Instagram performance intelligence</div>
            <h1 class="hero-title">Know the <span class="accent">signal</span><br>before you <span class="outline">publish.</span></h1>
            <div class="hero-copy">
                A machine-learning prototype that estimates the raw engagement of a planned Instagram post from account history, content choices, and publishing context.
            </div>
            <div class="hero-actions">
                <a class="cta-primary" href="#predictor">Run a prediction&nbsp; ↘</a>
                <a class="cta-secondary" href="#overview">Explore the project</a>
            </div>
        </div>
        <div class="hero-rail">
            <div class="signal-card">
                <div class="signal-head">
                    <div class="metric-kicker">Model signal / 01</div>
                    <span class="live-dot"></span>
                </div>
                <div class="signal-big">{int(meta['n_trees']):,} trees</div>
                <div class="signal-caption">Compact Random Forest inference loaded directly from the deployment artifact.</div>
                <div class="signal-bars">
                    <i style="height:28%"></i><i style="height:43%"></i><i style="height:35%"></i><i style="height:58%"></i>
                    <i style="height:82%"></i><i style="height:61%"></i><i style="height:48%"></i><i style="height:91%"></i>
                    <i style="height:70%"></i><i style="height:53%"></i><i style="height:64%"></i><i style="height:76%"></i>
                </div>
                <div class="signal-grid">
                    <div class="signal-mini"><span>TEST R²</span><strong>{meta['metrics']['r2']:.3f}</strong></div>
                    <div class="signal-mini"><span>TEST MAE</span><strong>{meta['metrics']['mae']:,.0f}</strong></div>
                </div>
            </div>
            <div class="signal-card compact">
                <div class="metric-kicker">Inputs / 02</div>
                <div class="signal-grid">
                    <div class="signal-mini"><span>ACCOUNT</span><strong>3 signals</strong></div>
                    <div class="signal-mini"><span>CONTENT</span><strong>4 signals</strong></div>
                    <div class="signal-mini"><span>TIMING</span><strong>derived</strong></div>
                    <div class="signal-mini"><span>OUTPUT</span><strong>engagement</strong></div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Project overview section
overview_html = """
<div id="overview" class="section-anchor"></div>
<section class="light-band about-section">
<div class="about-head">
<div class="about-index">01 / PROJECT OVERVIEW</div>
<div class="about-head-main">
<h2 class="about-title">Plan with evidence.<br>Publish with context.</h2>
<p class="about-intro">The project estimates the raw engagement of a planned Instagram post from information available before publishing: account history, content format, copy, hashtags, date, and time.</p>
</div>
</div>
<div class="about-thesis">
<div class="about-quote">
<div class="about-quote-kicker">The idea</div>
<p>Turn <em>post planning</em> into a measurable machine-learning problem instead of relying only on intuition after the post is already live.</p>
</div>
<div class="about-summary">
<div class="about-summary-item">
<div class="about-summary-no">A /</div>
<div><strong>Start with the account</strong><span>Follower count and historical engagement establish the account baseline.</span></div>
</div>
<div class="about-summary-item">
<div class="about-summary-no">B /</div>
<div><strong>Describe the planned post</strong><span>Media type, caption, hashtags, date, and time become structured model inputs.</span></div>
</div>
<div class="about-summary-item">
<div class="about-summary-no">C /</div>
<div><strong>Read the estimate in context</strong><span>The output is compared with historical performance as a planning signal, not a guaranteed result.</span></div>
</div>
</div>
</div>
<div class="about-pipeline-head">
<strong>From historical data to a prediction</strong>
<span>Four-stage inference flow</span>
</div>
<div class="architecture-strip">
<div class="architecture-node"><span class="micro-label">01 · Source</span><strong>Historical Instagram data</strong></div>
<div class="architecture-node"><span class="micro-label">02 · Transform</span><strong>Feature engineering</strong></div>
<div class="architecture-node"><span class="micro-label">03 · Model</span><strong>Random Forest inference</strong></div>
<div class="architecture-node"><span class="micro-label">04 · Output</span><strong>Expected engagement</strong></div>
</div>
</section>
"""
# Strip leading whitespace/newlines so Streamlit's Markdown parser cannot
# reinterpret nested HTML as an indented code block after a blank line.
st.markdown("".join(line.strip() for line in overview_html.splitlines()), unsafe_allow_html=True)

# Predictor section
st.markdown(
    """
    <div id="predictor" class="section-anchor"></div>
    <section class="dark-band" style="padding-bottom:2rem;">
        <div class="predictor-intro">
            <span class="predictor-kicker">02 / INTERACTIVE PREDICTOR</span>
            <h2 class="predictor-title">Build the post.<br>Read the signal.</h2>
            <div class="predictor-sub">Give the model the same information a social-media team would know before publishing: account history, content format, copy length, hashtags, date, and time.</div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

account_col, post_col = st.columns([0.9, 1.4], gap="large")

with account_col:
    with st.container(border=True):
        st.markdown("### Account baseline")
        st.caption("Context from the account's existing performance.")

        user_median_engagement = st.number_input(
            "Historical median engagement",
            min_value=0.0,
            value=float(meta["default_user_median_engagement"]),
            step=1.0,
            help="Median raw engagement from previous posts of the account.",
        )

        user_post_count = st.number_input(
            "Number of prior posts",
            min_value=0,
            value=int(meta["default_user_post_count"]),
            step=1,
        )

        followers = st.number_input(
            "Current followers",
            min_value=1,
            value=10000,
            step=100,
        )

with post_col:
    with st.container(border=True):
        st.markdown("### Planned post")
        st.caption("Describe the content you are preparing to publish.")

        p1, p2 = st.columns(2, gap="medium")
        with p1:
            media_choice = st.selectbox(
                "Media type",
                ["Image", "Carousel", "Video / Reel"],
            )

            number_hashtags = st.number_input(
                "Number of hashtags",
                min_value=0,
                max_value=30,
                value=5,
                step=1,
            )

            post_date = st.date_input("Planned post date", value=date.today())

        with p2:
            length_caption = st.number_input(
                "Caption length",
                min_value=0,
                max_value=10000,
                value=120,
                step=1,
                help="Number of characters in the planned caption.",
            )

            post_hour = st.slider(
                "Publishing hour",
                min_value=0,
                max_value=23,
                value=12,
                help="Hour of day using a 24-hour clock.",
            )

            if media_choice == "Carousel":
                post_images = st.number_input(
                    "Images in carousel",
                    min_value=2,
                    max_value=30,
                    value=5,
                    step=1,
                )
            elif media_choice == "Image":
                post_images = 1
                st.caption("Single-image post · image count = 1")
            else:
                post_images = 0
                st.caption("Video / Reel · image count = 0")

# Derived features
weekday = post_date.strftime("%A")
is_weekend = int(weekday in ["Saturday", "Sunday"])
is_holiday = int(post_date in us_holidays(post_date.year))
reach_time_bucket = bucket(post_hour)

if length_caption <= 50:
    caption_length_bucket = "short"
elif length_caption <= 150:
    caption_length_bucket = "medium"
elif length_caption <= 300:
    caption_length_bucket = "long"
else:
    caption_length_bucket = "very_long"

if number_hashtags == 0:
    hashtag_bucket = "none"
elif number_hashtags <= 5:
    hashtag_bucket = "low"
elif number_hashtags <= 15:
    hashtag_bucket = "medium"
else:
    hashtag_bucket = "high"

holiday_label = "Holiday" if is_holiday else "Non-holiday"
weekend_label = "Weekend" if is_weekend else "Weekday"

st.markdown(
    f"""
    <div class="auto-strip">
        <span class="auto-label">Auto-derived</span>
        <span class="feature-tag">{weekday}</span>
        <span class="feature-tag">{weekend_label}</span>
        <span class="feature-tag">{holiday_label}</span>
        <span class="feature-tag">{reach_time_bucket.upper()} TIME</span>
        <span class="feature-tag">{caption_length_bucket.replace('_', ' ').upper()} CAPTION</span>
        <span class="feature-tag">{hashtag_bucket.upper()} HASHTAGS</span>
    </div>
    """,
    unsafe_allow_html=True,
)

predict_clicked = st.button(
    "RUN ENGAGEMENT PREDICTION  →",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    x = make_features(
        followers=followers,
        post_images=post_images,
        video=int(media_choice == "Video / Reel"),
        carousel=int(media_choice == "Carousel"),
        length_caption=length_caption,
        number_hashtags=number_hashtags,
        publication_weekday=weekday,
        is_weekend=is_weekend,
        is_holiday=is_holiday,
        user_median_engagement=user_median_engagement,
        user_post_count=user_post_count,
        reach_time_bucket=reach_time_bucket,
        caption_length_bucket=caption_length_bucket,
        hashtag_bucket=hashtag_bucket,
    )

    prediction = predict_one(x)

    if user_median_engagement > 0:
        change = (prediction / user_median_engagement - 1) * 100
        if change > 0.05:
            delta_class = "positive"
            delta_text = f"UP {abs(change):.1f}% VS HISTORICAL MEDIAN"
        elif change < -0.05:
            delta_class = "negative"
            delta_text = f"DOWN {abs(change):.1f}% VS HISTORICAL MEDIAN"
        else:
            delta_class = ""
            delta_text = "IN LINE WITH HISTORICAL MEDIAN"
    else:
        change = None
        delta_class = ""
        delta_text = "NO HISTORICAL MEDIAN ENTERED"

    max_value = max(float(prediction), float(user_median_engagement), 1.0)
    prediction_width = max(3.0, min(100.0, float(prediction) / max_value * 100.0))
    median_width = max(3.0, min(100.0, float(user_median_engagement) / max_value * 100.0))

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    result_col, benchmark_col = st.columns([1.25, 0.85], gap="large")

    with result_col:
        st.markdown(
            f"""
            <div class="result-wrap">
                <div class="result-kicker">Prediction result / raw engagement</div>
                <div class="result-value">{prediction:,.0f}</div>
                <div class="result-copy">Estimated total engagement for the planned post based on the account context and post characteristics entered above.</div>
                <div class="result-delta {delta_class}">{delta_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with benchmark_col:
        st.markdown(
            f"""
            <div class="benchmark-box">
                <div class="benchmark-title">Benchmark view</div>
                <div class="compare-row">
                    <div class="compare-top"><span>Prediction</span><strong>{prediction:,.0f}</strong></div>
                    <div class="compare-track"><div class="compare-fill" style="width:{prediction_width:.1f}%"></div></div>
                </div>
                <div class="compare-row">
                    <div class="compare-top"><span>Historical median</span><strong>{user_median_engagement:,.0f}</strong></div>
                    <div class="compare-track"><div class="compare-fill median" style="width:{median_width:.1f}%"></div></div>
                </div>
                <div class="compare-row">
                    <div class="compare-top"><span>Prior posts</span><strong>{user_post_count:,}</strong></div>
                </div>
                <div class="result-note">This output is a model estimate. Actual Instagram performance can differ because the model does not observe every creative, audience, platform, or distribution factor.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Technical model details"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Trees", f"{int(meta['n_trees']):,}")
        d2.metric("Test R²", f"{meta['metrics']['r2']:.3f}")
        d3.metric("Test RMSE", f"{meta['metrics']['rmse']:,.0f}")
        d4.metric("Test MAE", f"{meta['metrics']['mae']:,.0f}")
        st.caption("Inference uses the compact model arrays in model.npz and the feature metadata in model_meta.json.")

# Method section
st.markdown(
    """
    <div id="method" class="section-anchor"></div>
    <section class="light-band" style="margin-top:4.5rem;">
        <div class="section-heading">
            <div class="section-index">03 / METHOD</div>
            <div>
                <h2 class="section-title">From post idea to model-ready features.</h2>
                <div class="section-copy">The interface hides the preprocessing complexity. It collects simple planning inputs, derives categorical signals, builds the feature vector, and evaluates the trained forest.</div>
            </div>
        </div>
        <div class="method-grid">
            <div class="method-card"><span class="step-index">STEP 01</span><span class="arrow">↘</span><h4>Enter planning context</h4><p>Followers, historical engagement, prior posts, media type, caption, hashtags, date, and hour.</p></div>
            <div class="method-card"><span class="step-index">STEP 02</span><span class="arrow">↘</span><h4>Derive categories</h4><p>The app detects weekday, weekend, holiday, time bucket, caption bucket, and hashtag bucket.</p></div>
            <div class="method-card"><span class="step-index">STEP 03</span><span class="arrow">↘</span><h4>Evaluate the forest</h4><p>The feature vector moves through each stored decision tree and reaches a leaf value.</p></div>
            <div class="method-card"><span class="step-index">STEP 04</span><span class="arrow">●</span><h4>Average the outputs</h4><p>Tree outputs are averaged to produce the final expected raw engagement estimate.</p></div>
        </div>
        <div class="feature-matrix">
            <div class="feature-cell"><h5>Account signals</h5><p>followers · user_median_engagement · user_post_count</p></div>
            <div class="feature-cell"><h5>Content signals</h5><p>media type · post_images · length_caption · number_hashtags</p></div>
            <div class="feature-cell"><h5>Timing signals</h5><p>publication_weekday · weekend · holiday · reach_time_bucket</p></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Model section
st.markdown(
    f"""
    <div id="model" class="section-anchor"></div>
    <section class="dark-band">
        <div class="section-heading">
            <div class="section-index">04 / MODEL</div>
            <div>
                <h2 class="section-title">Lightweight deployment.<br>No pickle dependency.</h2>
            </div>
        </div>
        <div class="model-panel">
            <div>
                <p class="model-copy-large">The deployed app reads the trained forest from a compact <span style="color:var(--orange)">NPZ model artifact</span> and evaluates the trees directly.</p>
                <div class="model-body">That keeps the prediction runtime small and separates the training environment from the deployment interface. Model metadata, feature names, categorical values, defaults, and evaluation metrics are read from the accompanying JSON file.</div>
                <div class="disclaimer"><strong style="color:#DAD7CF">Interpretation note.</strong> R², RMSE, and MAE describe evaluation on the model's test split. They do not guarantee the performance of a future Instagram post.</div>
            </div>
            <div class="metric-stack">
                <div class="metric-line"><span>RANDOM FOREST TREES</span><strong>{int(meta['n_trees']):,}</strong></div>
                <div class="metric-line"><span>TEST R²</span><strong>{meta['metrics']['r2']:.3f}</strong></div>
                <div class="metric-line"><span>TEST RMSE</span><strong>{meta['metrics']['rmse']:,.0f}</strong></div>
                <div class="metric-line"><span>TEST MAE</span><strong>{meta['metrics']['mae']:,.0f}</strong></div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Final About / Team section
team_cards = []
for member in TEAM_MEMBERS:
    image_source = member.get("image", "")
    image_url = image_to_data_uri(image_source) if image_source else ""
    name = member.get("name", "Team member")
    linkedin = member.get("linkedin", "")
    github = member.get("github", "")

    links = []
    if linkedin:
        links.append(f'<a class="team-link" href="{linkedin}" target="_blank" rel="noopener noreferrer">LinkedIn ↗</a>')
    if github:
        links.append(f'<a class="team-link" href="{github}" target="_blank" rel="noopener noreferrer">GitHub ↗</a>')

    initials = "".join(part[0].upper() for part in name.split()[:2] if part)
    image_html = (
        f'<img class="team-photo" src="{image_url}" alt="{name}">'
        if image_url
        else f'<div class="team-initials" aria-label="{name}">{initials}</div>'
    )
    team_cards.append(
        f'<article class="team-card">{image_html}<div class="team-card-body"><div class="team-name">{name}</div><div class="team-links">{"".join(links)}</div></div></article>'
    )

if team_cards:
    team_grid_html = f'<div class="team-grid">{"".join(team_cards)}</div>'
else:
    team_grid_html = "<div class='team-empty'>Team profiles are ready to be added here. Add each member's name, role, image URL, LinkedIn URL, and GitHub URL in the TEAM_MEMBERS list near the top of this file.</div>"

about_team_html = f"""
<div id="about" class="section-anchor"></div>
<section class="team-band">
<div class="team-head">
<div class="team-index">05 / ABOUT</div>
<div>
<h2 class="team-title">The people behind the project.</h2>
<p class="team-intro">Meet the team behind the Instagram Engagement Predictor. Open a profile to connect on LinkedIn or explore their GitHub when available.</p>
</div>
</div>
{team_grid_html}
</section>
"""
st.markdown(about_team_html, unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <footer class="site-footer">
        <div>
            <div class="footer-title">Instagram Engagement Predictor</div>
            <div class="footer-copy">Target: raw engagement — likes + comments, with shares and saves where those fields are available in the source data.</div>
        </div>
        <div class="footer-meta">ML PROJECT / STREAMLIT DEMO<br>DESIGNED AS AN EDITORIAL DATA PRODUCT</div>
    </footer>
    """,
    unsafe_allow_html=True,
)
