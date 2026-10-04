import re
import string

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from gensim.models import Word2Vec
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.decomposition import PCA

st.set_page_config(page_title="FitText Lab", page_icon="⚡", layout="wide")

NEON = "#c6ff00"
PINK = "#ff2e88"
CYAN = "#00e5ff"


# ---------- styling ----------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@400;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif; }}
.stApp {{
    background: radial-gradient(circle at 15% 10%, #1d1f33 0%, #0a0a12 45%, #050508 100%);
    color: #e9e9f2;
}}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.hero {{
    padding: 2.2rem 2rem 1.6rem;
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(198,255,0,.08), rgba(255,46,136,.08));
    border: 1px solid rgba(255,255,255,.08);
    position: relative;
    overflow: hidden;
    margin-bottom: 1.4rem;
}}
.hero:before {{
    content: "";
    position: absolute; inset: -40%;
    background: conic-gradient(from 0deg, transparent, {NEON}22, transparent 30%);
    animation: spin 9s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.hero * {{ position: relative; }}
.hero h1 {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 8vw, 6rem);
    line-height: .9;
    margin: 0;
    letter-spacing: 2px;
    background: linear-gradient(90deg, {NEON}, {CYAN}, {PINK});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.hero p {{ color: #a7a7bd; font-size: 1.05rem; margin: .6rem 0 0; }}

.stat {{
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    transition: transform .2s, border-color .2s;
}}
.stat:hover {{ transform: translateY(-4px); border-color: {NEON}; }}
.stat .num {{ font-family: 'Bebas Neue'; font-size: 2.6rem; color: {NEON}; line-height: 1; }}
.stat .lbl {{ color: #8f8fa8; font-size: .85rem; text-transform: uppercase; letter-spacing: 1px; }}

.chip {{
    display: inline-block;
    padding: .28rem .7rem;
    margin: .18rem;
    border-radius: 999px;
    font-size: .88rem;
    background: rgba(0,229,255,.1);
    border: 1px solid rgba(0,229,255,.35);
    color: {CYAN};
}}
.chip.gone {{ background: rgba(255,46,136,.08); border-color: rgba(255,46,136,.35); color: {PINK}; text-decoration: line-through; }}
.chip.lime {{ background: rgba(198,255,0,.08); border-color: rgba(198,255,0,.4); color: {NEON}; }}

.step {{ font-family: 'Bebas Neue'; font-size: 1.5rem; color: #fff; letter-spacing: 1px; margin-top: 1rem; }}
.step span {{ color: {PINK}; margin-right: .4rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: .5rem; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,.04);
    border-radius: 12px;
    padding: .5rem 1.1rem;
    color: #bbb;
}}
.stTabs [aria-selected="true"] {{ background: {NEON} !important; color: #000 !important; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- data + models ----------
@st.cache_resource
def get_nltk():
    needed = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet.zip": "wordnet",
        "corpora/omw-1.4.zip": "omw-1.4",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    }
    for path, pkg in needed.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    return True


get_nltk()
STOP = set(stopwords.words("english"))
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
regex_tok = RegexpTokenizer(r"[a-zA-Z]+")


@st.cache_data
def load_data():
    corpus = pd.read_csv("data/word2vec_corpus.csv")["text"].tolist()
    raw = pd.read_csv("data/dataset.csv")["text"].tolist()
    return corpus, raw


def w2v_tokens(sentence):
    s = sentence.lower().replace("-", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    return [w for w in word_tokenize(s) if w not in STOP]


@st.cache_resource
def train(vector_size, window, epochs, sg, seed):
    corpus, _ = load_data()
    toks = [w2v_tokens(s) for s in corpus]
    model = Word2Vec(toks, vector_size=vector_size, window=window, min_count=1,
                     sg=sg, epochs=epochs, seed=seed, workers=1)
    return model, toks


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wn_pos(tag):
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def lemmatize(tokens):
    return [lemmatizer.lemmatize(w, wn_pos(t)) for w, t in nltk.pos_tag(tokens)]


def chips(words, kind=""):
    return "".join(f'<span class="chip {kind}">{w}</span>' for w in words)


def dark(fig, h=520):
    fig.update_layout(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d8d8e6", family="Space Grotesk"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


corpus, raw_sentences = load_data()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown(f"<h2 style='font-family:Bebas Neue;color:{NEON};letter-spacing:2px'>MODEL KNOBS</h2>",
                unsafe_allow_html=True)
    vector_size = st.slider("vector_size", 10, 100, 50, 10)
    window = st.slider("window", 1, 8, 3)
    epochs = st.slider("epochs", 50, 1000, 500, 50)
    sg = 1 if st.radio("algorithm", ["Skip-gram", "CBOW"]) == "Skip-gram" else 0
    seed = st.number_input("seed", 0, 9999, 42)
    st.caption("Change anything and the model retrains instantly — tiny data, tiny wait.")

model, w2v_toks = train(vector_size, window, epochs, sg, seed)
vocab = model.wv.index_to_key

# ---------- hero ----------
st.markdown(
    """
<div class="hero">
  <h1>FIT·TEXT LAB</h1>
  <p>Word2Vec embeddings + a full NLP preprocessing pipeline, trained on health & fitness sentences.
  Poke the knobs on the left and watch the word space move.</p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in [
    (c1, len(corpus), "W2V sentences"),
    (c2, len(vocab), "vocab learned"),
    (c3, vector_size, "dimensions"),
    (c4, len(raw_sentences), "noisy sentences"),
]:
    col.markdown(f'<div class="stat"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["🌌  Embedding Space", "🎯  Similarity", "🧪  Preprocessing Lab", "⚔️  Stemmer Showdown"])

# ---------- tab 1: word space ----------
with tab1:
    vecs = np.array([model.wv[w] for w in vocab])
    pts = PCA(n_components=2, random_state=0).fit_transform(vecs)
    counts = [model.wv.get_vecattr(w, "count") for w in vocab]

    focus = st.selectbox("Highlight a word and its neighbours", vocab, index=vocab.index("diet"))
    near = {w for w, _ in model.wv.most_similar(focus, topn=5)}
    group = ["focus" if w == focus else "neighbour" if w in near else "other" for w in vocab]

    emb = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "word": vocab,
                        "count": counts, "group": group})
    # only label the interesting words, otherwise 100 labels pile on top of each other
    emb["label"] = np.where((emb.group != "other") | (emb["count"] >= 2), emb.word, "")
    fig = px.scatter(
        emb, x="x", y="y", text="label", hover_name="word", color="group", size="count", size_max=34,
        color_discrete_map={"focus": PINK, "neighbour": NEON, "other": "#5b5b7a"},
        hover_data={"x": False, "y": False, "label": False, "count": True},
    )
    fig.update_traces(textposition="top center", textfont=dict(size=11),
                      marker=dict(line=dict(width=0), opacity=.9))
    # lines from the focus word to its neighbours
    fx, fy = emb.loc[emb.word == focus, ["x", "y"]].values[0]
    for _, r in emb[emb.group == "neighbour"].iterrows():
        fig.add_shape(type="line", x0=fx, y0=fy, x1=r.x, y1=r.y,
                      line=dict(color=NEON, width=1, dash="dot"), layer="below")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend=dict(orientation="h", y=1.08, title=None))
    st.plotly_chart(dark(fig, 620), width="stretch")
    st.caption("Every word squeezed from its vector down to 2D with PCA. Bigger dot = word appears more often. "
               "Note: neighbours are found in the full vector space, so they may not look closest in 2D.")

# ---------- tab 2: similarity ----------
with tab2:
    left, right = st.columns(2)
    for col, default in [(left, "diet"), (right, "workout")]:
        with col:
            word = st.selectbox("Word", vocab, index=vocab.index(default), key=f"sim_{default}")
            sims = pd.DataFrame(model.wv.most_similar(word, topn=5), columns=["word", "similarity"])
            bar = go.Figure(go.Bar(
                x=sims.similarity[::-1], y=sims.word[::-1], orientation="h",
                marker=dict(color=sims.similarity[::-1], colorscale=[[0, CYAN], [1, NEON]]),
                text=sims.similarity[::-1].round(3), textposition="outside",
            ))
            bar.update_xaxes(range=[0, 1.1], showgrid=False)
            st.plotly_chart(dark(bar, 330), width="stretch")
            with st.expander(f'vector for "{word}"'):
                v = model.wv[word]
                st.write(f"shape: {v.shape}")
                heat = go.Figure(go.Heatmap(z=[v], colorscale=[[0, PINK], [.5, "#111"], [1, NEON]],
                                            showscale=False))
                heat.update_yaxes(visible=False)
                st.plotly_chart(dark(heat, 120), width="stretch")
                st.code(np.round(v[:10], 4))

    st.markdown('<div class="step"><span>//</span>Word math</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    pos1 = a.selectbox("start with", vocab, index=vocab.index("workout"))
    neg = b.selectbox("minus", vocab, index=vocab.index("strength"))
    pos2 = c.selectbox("plus", vocab, index=vocab.index("diet"))
    res = model.wv.most_similar(positive=[pos1, pos2], negative=[neg], topn=5)
    st.markdown(chips([f"{w} · {s:.2f}" for w, s in res], "lime"), unsafe_allow_html=True)
    st.caption("Honestly, with 20 sentences this is mostly noise — that's the point. Analogies need big data.")

# ---------- tab 3: preprocessing ----------
with tab3:
    pick = st.selectbox("Pick a noisy sentence (or type your own below)", raw_sentences)
    text = st.text_input("Raw text", pick)

    cleaned = clean_text(text)
    wt = word_tokenize(text)
    rt = regex_tok.tokenize(cleaned)
    kept = [w for w in rt if w not in STOP]

    st.markdown('<div class="step"><span>01</span>Clean</div>', unsafe_allow_html=True)
    st.markdown(f"`{text}`  →  `{cleaned}`")

    st.markdown('<div class="step"><span>02</span>Tokenize</div>', unsafe_allow_html=True)
    st.markdown("**word_tokenize (raw)** " + chips(wt), unsafe_allow_html=True)
    st.markdown("**RegexpTokenizer (clean)** " + chips(rt), unsafe_allow_html=True)

    st.markdown('<div class="step"><span>03</span>Stopwords</div>', unsafe_allow_html=True)
    st.markdown("".join(chips([w], "gone" if w in STOP else "") for w in rt), unsafe_allow_html=True)
    st.caption(f"{len(rt)} tokens → {len(kept)} tokens")

    st.markdown('<div class="step"><span>04</span>Stem & Lemmatize</div>', unsafe_allow_html=True)
    if kept:
        st.dataframe(pd.DataFrame({
            "token": kept,
            "porter": [porter.stem(w) for w in kept],
            "snowball": [snowball.stem(w) for w in kept],
            "lemma": lemmatize(kept),
        }), width="stretch", hide_index=True)

    # vocab funnel over the whole dataset
    st.markdown('<div class="step"><span>05</span>Vocabulary funnel (whole dataset)</div>',
                unsafe_allow_html=True)
    all_tok = [regex_tok.tokenize(clean_text(s)) for s in raw_sentences]
    all_kept = [[w for w in t if w not in STOP] for t in all_tok]
    stages = {
        "Raw tokens": {w for t in all_tok for w in t},
        "No stopwords": {w for t in all_kept for w in t},
        "Porter": {porter.stem(w) for t in all_kept for w in t},
        "Snowball": {snowball.stem(w) for t in all_kept for w in t},
        "Lemmatized": {w for t in all_kept for w in lemmatize(t)},
    }
    funnel = go.Figure(go.Funnel(
        y=list(stages), x=[len(v) for v in stages.values()],
        marker=dict(color=[PINK, "#b04dff", CYAN, "#3ddc97", NEON]),
        textinfo="value",
    ))
    st.plotly_chart(dark(funnel, 360), width="stretch")

# ---------- tab 4: porter vs snowball ----------
with tab4:
    all_words = sorted({w for s in raw_sentences for w in regex_tok.tokenize(clean_text(s)) if w not in STOP})
    table = pd.DataFrame({
        "Original": all_words,
        "Porter": [porter.stem(w) for w in all_words],
        "Snowball": [snowball.stem(w) for w in all_words],
        "Lemma": [lemmatizer.lemmatize(w, wn_pos(t)) for w, t in nltk.pos_tag(all_words)],
    })
    table["Clash"] = np.where(table.Porter != table.Snowball, "⚡", "")

    only_clash = st.toggle("Show only words where Porter and Snowball disagree")
    view = table[table.Clash == "⚡"] if only_clash else table

    st.dataframe(
        view.style.apply(lambda r: [f"background-color: {PINK}33" if r.Clash else ""] * len(r), axis=1),
        width="stretch", hide_index=True, height=480,
    )

    st.markdown('<div class="step"><span>//</span>Try any word</div>', unsafe_allow_html=True)
    w = st.text_input("word", "generously").strip().lower()
    if w:
        k1, k2, k3 = st.columns(3)
        for col, lbl, val in [(k1, "Porter", porter.stem(w)), (k2, "Snowball", snowball.stem(w)),
                              (k3, "Lemma (verb)", lemmatizer.lemmatize(w, "v"))]:
            col.markdown(f'<div class="stat"><div class="num">{val}</div><div class="lbl">{lbl}</div></div>',
                         unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;color:#555;margin-top:3rem'>Text Analytics · Set D · built with Gensim, NLTK & Streamlit</p>",
    unsafe_allow_html=True,
)
