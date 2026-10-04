# FitText Lab ⚡

Text Analytics assignment (Set D) – Word2Vec + a text preprocessing pipeline on health & fitness sentences.

| File | What it is |
|---|---|
| `TextAnalytics_SetD.ipynb` | the main notebook, Q1 and Q2 with all outputs |
| `data/word2vec_corpus.csv` | 20 sentences used for Word2Vec (Q1) |
| `data/dataset.csv` | 12 noisy sentences used for preprocessing (Q2) |
| `docs/Project_Explanation.docx` | write-up with approach, outputs and answers |
| `app.py` | Streamlit app to play with the model and the pipeline |

## Q1 – Word2Vec
clean → tokenize → remove stopwords → train Gensim `Word2Vec(vector_size=50, window=3, min_count=1, sg=1, epochs=500)`
→ look at vocab, the vector for **diet**, and top-5 similar words for **diet** and **workout**.

## Q2 – Preprocessing
clean (regex) → `word_tokenize` vs `RegexpTokenizer` → stopwords → Porter / Snowball stemming → WordNet lemmatization (with POS tags)
→ comparison table + vocab size at every stage.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the notebook: `pip install jupyter` and open `TextAnalytics_SetD.ipynb`.

## Deploy
Push to GitHub → go to [share.streamlit.io](https://share.streamlit.io) → New app → pick this repo, branch `main`, file `app.py`. Done.

## App tabs
- **Embedding Space** – every word plotted in 2D (PCA), pick a word to light up its neighbours
- **Similarity** – top-5 bars, the raw vector as a heatmap, and some (very noisy) word math
- **Preprocessing Lab** – type any sentence and watch it go through each step
- **Stemmer Showdown** – Porter vs Snowball vs Lemma for every word, clashes highlighted

Sidebar sliders retrain the model live, so you can see what `window`, `epochs` etc. actually do.
