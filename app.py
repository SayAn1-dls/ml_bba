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

st.set_page_config(page_title="Health & Fitness Text Analytics", page_icon=":material/analytics:", layout="wide")

INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
ACCENT = "#1d4ed8"
TEAL = "#0f766e"
AMBER = "#b45309"
GREY = "#cbd5e1"


# ---------- styling ----------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, p, label {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: #f8fafc; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1200px; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.eyebrow {{ font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: {ACCENT}; }}
.title {{ font-size: 2.1rem; font-weight: 700; color: {INK}; margin: .25rem 0 .4rem; letter-spacing: -.02em; }}
.subtitle {{ color: {MUTED}; font-size: 1rem; max-width: 760px; line-height: 1.55; }}
.rule {{ border: 0; border-top: 1px solid {LINE}; margin: 1.4rem 0 1.2rem; }}

.metric {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: .9rem 1.1rem; }}
.metric .v {{ font-size: 1.7rem; font-weight: 600; color: {INK}; line-height: 1.1; }}
.metric .l {{ font-size: .8rem; color: {MUTED}; margin-top: .2rem; }}

.section {{ font-size: 1.05rem; font-weight: 600; color: {INK}; margin: 1.4rem 0 .2rem; }}
.note {{ font-size: .88rem; color: {MUTED}; margin-bottom: .6rem; }}

.card {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 1rem 1.2rem; height: 100%; }}
.card h4 {{ margin: 0 0 .35rem; font-size: .95rem; color: {INK}; }}
.card p {{ margin: 0; font-size: .88rem; color: {MUTED}; line-height: 1.5; }}

.steps {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .4rem 0 .2rem; }}
.steps .s {{ background: #fff; border: 1px solid {LINE}; border-radius: 6px; padding: .35rem .7rem; font-size: .85rem; color: {INK}; }}
.steps .a {{ color: {GREY}; }}

.tok {{ display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: .8rem;
        padding: .15rem .5rem; margin: .15rem .2rem .15rem 0; border-radius: 4px;
        background: #f1f5f9; border: 1px solid {LINE}; color: {INK}; }}
.tok.drop {{ background: #fef2f2; border-color: #fecaca; color: #b91c1c; text-decoration: line-through; }}
.tok.hit {{ background: #eff6ff; border-color: #bfdbfe; color: {ACCENT}; }}
.row-label {{ font-size: .8rem; color: {MUTED}; margin-top: .6rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: .6rem 0; font-weight: 500; color: {MUTED}; background: transparent; }}
.stTabs [aria-selected="true"] {{ color: {INK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT}; }}

section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid {LINE}; }}
.footer {{ text-align: center; color: {MUTED}; font-size: .8rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid {LINE}; }}
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


def tokens_html(words, kind=""):
    return "".join(f'<span class="tok {kind}">{w}</span>' for w in words)


def section(title, note=None):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)


def style_fig(fig, h=420):
    fig.update_layout(
        template="plotly_white",
        height=h,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


corpus, raw_sentences = load_data()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### Word2Vec parameters")
    st.caption("Defaults are the same values used in the notebook.")
    vector_size = st.slider("Vector size", 10, 100, 50, 10)
    window = st.slider("Window", 1, 8, 3)
    epochs = st.slider("Epochs", 50, 1000, 500, 50)
    sg = 1 if st.radio("Algorithm", ["Skip-gram", "CBOW"], horizontal=True) == "Skip-gram" else 0
    seed = st.number_input("Random seed", 0, 9999, 42)
    st.divider()
    st.caption("min_count is fixed at 1 because almost every word appears only once in this corpus.")

model, w2v_toks = train(vector_size, window, epochs, sg, seed)
vocab = model.wv.index_to_key

# ---------- header ----------
st.markdown(
    """
<div class="eyebrow">Text Analytics · Assignment Set D</div>
<div class="title">Health &amp; Fitness Text Analytics</div>
<div class="subtitle">Word embeddings with Gensim Word2Vec and a complete NLTK preprocessing pipeline —
cleaning, tokenization, stopword removal, stemming and lemmatization — applied to short health and fitness sentences.</div>
""",
    unsafe_allow_html=True,
)
st.write("")

cols = st.columns(4)
for col, v, l in [
    (cols[0], len(corpus), "Word2Vec sentences"),
    (cols[1], len(vocab), "Vocabulary size"),
    (cols[2], vector_size, "Vector dimensions"),
    (cols[3], len(raw_sentences), "Noisy sentences"),
]:
    col.markdown(f'<div class="metric"><div class="v">{v}</div><div class="l">{l}</div></div>',
                 unsafe_allow_html=True)

st.write("")
t0, t1, t2, t3, t4 = st.tabs(["Overview", "Word Embeddings", "Similarity", "Preprocessing", "Stemming vs Lemmatization"])

# ---------- overview ----------
with t0:
    section("Problem statement")
    a, b = st.columns(2)
    a.markdown(
        """<div class="card"><h4>Q1 · Word2Vec</h4><p>Train a Word2Vec model on 20 health &amp; fitness
        sentences, inspect the vocabulary and word vectors, and find the words most similar to
        <b>diet</b> and <b>workout</b>. Then judge whether the similarities make sense on such a small corpus.</p></div>""",
        unsafe_allow_html=True,
    )
    b.markdown(
        """<div class="card"><h4>Q2 · Preprocessing pipeline</h4><p>Take 12 noisy sentences (mixed case,
        punctuation, numbers, emoticons) and clean them for NLP. Compare two tokenizers, two stemmers and
        lemmatization, and track how the vocabulary shrinks at each step.</p></div>""",
        unsafe_allow_html=True,
    )

    section("Pipeline")
    steps = ["Raw text", "Lowercase", "Remove punctuation / numbers", "Tokenize", "Remove stopwords",
             "Stem / Lemmatize", "Word2Vec / Analysis"]
    st.markdown('<div class="steps">' + '<span class="a">→</span>'.join(f'<span class="s">{s}</span>' for s in steps)
                + "</div>", unsafe_allow_html=True)

    section("Datasets")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="note">word2vec_corpus.csv — clean sentences for Q1</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"text": corpus}), hide_index=True, height=300, width="stretch")
    with b:
        st.markdown('<div class="note">dataset.csv — raw, noisy sentences for Q2</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"text": raw_sentences}), hide_index=True, height=300, width="stretch")

# ---------- word embeddings ----------
with t1:
    section("Embedding space",
            "Each word's vector reduced to two dimensions with PCA. Marker size reflects how often the word appears.")
    vecs = np.array([model.wv[w] for w in vocab])
    pts = PCA(n_components=2, random_state=0).fit_transform(vecs)
    counts = [model.wv.get_vecattr(w, "count") for w in vocab]

    focus = st.selectbox("Highlight word", vocab, index=vocab.index("diet"))
    near = {w for w, _ in model.wv.most_similar(focus, topn=5)}
    group = ["Selected" if w == focus else "Top-5 similar" if w in near else "Other" for w in vocab]

    emb = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "word": vocab, "count": counts, "group": group})
    # only label the interesting words, otherwise 100 labels pile on top of each other
    emb["label"] = np.where((emb.group != "Other") | (emb["count"] >= 2), emb.word, "")
    fig = px.scatter(
        emb, x="x", y="y", text="label", hover_name="word", color="group", size="count", size_max=22,
        color_discrete_map={"Selected": AMBER, "Top-5 similar": ACCENT, "Other": GREY},
        category_orders={"group": ["Selected", "Top-5 similar", "Other"]},
        hover_data={"x": False, "y": False, "label": False, "count": True, "group": False},
    )
    fig.update_traces(textposition="top center", textfont=dict(size=11, color=INK), marker=dict(line=dict(width=0)))
    fx, fy = emb.loc[emb.word == focus, ["x", "y"]].values[0]
    for _, r in emb[emb.group == "Top-5 similar"].iterrows():
        fig.add_shape(type="line", x0=fx, y0=fy, x1=r.x, y1=r.y,
                      line=dict(color=ACCENT, width=1, dash="dot"), layer="below")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend=dict(orientation="h", y=1.06, x=0, title=None))
    st.plotly_chart(style_fig(fig, 560), width="stretch")
    st.caption("Similar words are found in the full vector space, so they may not look like the closest points in 2D.")

    section("Vocabulary", "All words learned by the model, most frequent first.")
    vocab_df = pd.DataFrame({"word": vocab, "count": counts})
    st.dataframe(vocab_df, hide_index=True, height=280, width="stretch")

# ---------- similarity ----------
with t2:
    section("Most similar words", "Cosine similarity between word vectors.")
    left, right = st.columns(2, gap="large")
    for col, default in [(left, "diet"), (right, "workout")]:
        with col:
            word = st.selectbox("Word", vocab, index=vocab.index(default), key=f"sim_{default}")
            sims = pd.DataFrame(model.wv.most_similar(word, topn=5), columns=["word", "similarity"])
            bar = go.Figure(go.Bar(
                x=sims.similarity[::-1], y=sims.word[::-1], orientation="h",
                marker_color=ACCENT, text=sims.similarity[::-1].round(3), textposition="outside",
            ))
            bar.update_xaxes(range=[0, 1.1], showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(style_fig(bar, 280), width="stretch")
            v = model.wv[word]
            with st.expander(f"Vector for '{word}'  ·  shape {v.shape}"):
                heat = go.Figure(go.Heatmap(z=[v], colorscale="RdBu", zmid=0, showscale=False))
                heat.update_yaxes(visible=False)
                st.plotly_chart(style_fig(heat, 110), width="stretch")
                st.code(np.round(v[:10], 4))

    section("Vector arithmetic",
            "A − B + C. With only 20 sentences the result is mostly noise — analogies need a large corpus.")
    a, b, c = st.columns(3)
    pos1 = a.selectbox("A", vocab, index=vocab.index("workout"))
    neg = b.selectbox("minus B", vocab, index=vocab.index("strength"))
    pos2 = c.selectbox("plus C", vocab, index=vocab.index("diet"))
    res = model.wv.most_similar(positive=[pos1, pos2], negative=[neg], topn=5)
    st.markdown(tokens_html([f"{w} ({s:.2f})" for w, s in res], "hit"), unsafe_allow_html=True)

# ---------- preprocessing ----------
with t3:
    section("Try a sentence", "Choose one of the dataset sentences or type your own.")
    pick = st.selectbox("Dataset sentence", raw_sentences, label_visibility="collapsed")
    text = st.text_input("Raw text", pick)

    cleaned = clean_text(text)
    wt = word_tokenize(text)
    rt = regex_tok.tokenize(cleaned)
    kept = [w for w in rt if w not in STOP]

    section("1. Cleaning")
    st.markdown(f'<div class="row-label">Before</div><code>{text}</code>'
                f'<div class="row-label">After</div><code>{cleaned}</code>', unsafe_allow_html=True)

    section("2. Tokenization")
    st.markdown('<div class="row-label">word_tokenize (on raw text)</div>' + tokens_html(wt), unsafe_allow_html=True)
    st.markdown('<div class="row-label">RegexpTokenizer [a-zA-Z]+ (on cleaned text)</div>' + tokens_html(rt),
                unsafe_allow_html=True)

    section("3. Stopword removal", f"{len(rt)} tokens → {len(kept)} tokens")
    st.markdown("".join(tokens_html([w], "drop" if w in STOP else "") for w in rt), unsafe_allow_html=True)

    section("4. Stemming and lemmatization")
    if kept:
        st.dataframe(pd.DataFrame({
            "Token": kept,
            "Porter": [porter.stem(w) for w in kept],
            "Snowball": [snowball.stem(w) for w in kept],
            "Lemma": lemmatize(kept),
        }), width="stretch", hide_index=True)

    section("Vocabulary size by stage", "Across all 12 sentences in dataset.csv.")
    all_tok = [regex_tok.tokenize(clean_text(s)) for s in raw_sentences]
    all_kept = [[w for w in t if w not in STOP] for t in all_tok]
    stages = {
        "Raw tokens": {w for t in all_tok for w in t},
        "After stopwords": {w for t in all_kept for w in t},
        "Porter": {porter.stem(w) for t in all_kept for w in t},
        "Snowball": {snowball.stem(w) for t in all_kept for w in t},
        "Lemmatized": {w for t in all_kept for w in lemmatize(t)},
    }
    vbar = go.Figure(go.Bar(
        x=list(stages), y=[len(v) for v in stages.values()],
        marker_color=[GREY, ACCENT, TEAL, TEAL, AMBER],
        text=[len(v) for v in stages.values()], textposition="outside",
    ))
    vbar.update_yaxes(range=[0, 90], gridcolor="#f1f5f9")
    st.plotly_chart(style_fig(vbar, 320), width="stretch")

# ---------- stemming vs lemmatization ----------
with t4:
    section("Comparison table", "Every distinct non-stopword in the dataset. Rows where Porter and Snowball disagree are highlighted.")
    all_words = sorted({w for s in raw_sentences for w in regex_tok.tokenize(clean_text(s)) if w not in STOP})
    table = pd.DataFrame({
        "Original": all_words,
        "Porter": [porter.stem(w) for w in all_words],
        "Snowball": [snowball.stem(w) for w in all_words],
        "Lemma": [lemmatizer.lemmatize(w, wn_pos(t)) for w, t in nltk.pos_tag(all_words)],
    })
    table["Disagree"] = table.Porter != table.Snowball

    only = st.toggle("Show only disagreements")
    view = table[table.Disagree] if only else table
    st.dataframe(
        view.style.apply(lambda r: ["background-color: #fef3c7" if r.Disagree else ""] * len(r), axis=1),
        width="stretch", hide_index=True, height=440,
    )

    section("Test any word")
    w = st.text_input("Word", "generously").strip().lower()
    if w:
        k = st.columns(3)
        for col, lbl, val in [(k[0], "Porter", porter.stem(w)), (k[1], "Snowball", snowball.stem(w)),
                              (k[2], "Lemma (as verb)", lemmatizer.lemmatize(w, "v"))]:
            col.markdown(f'<div class="metric"><div class="v">{val}</div><div class="l">{lbl}</div></div>',
                         unsafe_allow_html=True)

st.markdown('<div class="footer">Text Analytics · Set D · Python, Gensim, NLTK, Streamlit</div>',
            unsafe_allow_html=True)
