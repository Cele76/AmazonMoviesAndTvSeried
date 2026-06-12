# Prompt — Completamento Capitolo 1 (allineamento al report di riferimento)

## Riferimento
Repository: https://github.com/Cele76/AmazonMoviesAndTvSeried
File: `Progetto_MIGA_Completo_Movies (2).ipynb`

```bash
git clone https://github.com/Cele76/AmazonMoviesAndTvSeried.git
cd AmazonMoviesAndTvSeried
```

## Contesto

Il notebook attuale (Parte 1) già esegue: caricamento dati, distribuzione rating,
distribuzione temporale, recensioni per utente/film, pipeline di filtraggio a 4 step
(k-core asimmetrico utenti≥5 / film≥10 + top-K utenti), tabelle di distribuzione
per fasce, matrice di correlazione estesa.

Vanno AGGIUNTE le funzionalità del Capitolo 1 del report di riferimento che mancano.
NON riscrivere ciò che già esiste: inserisci nuove celle nei punti indicati.
Tutto in italiano, coerente con lo stile del notebook esistente.

---

## AGGIUNTA 1 — Tabella statistiche di base (inserire subito dopo la cella 11, pulizia colonne)

Crea una tabella riassuntiva del corpus originale prima di qualsiasi filtraggio:

```python
# Tabella 1.1 — Statistiche di base del dataset
stats_base = pd.DataFrame({
    "Metrica": ["Utenti unici", "Film unici (parent_asin)", "Recensioni totali"],
    "Valore": [
        df["user_id"].nunique(),
        df["item_id"].nunique(),
        len(df)
    ]
})
stats_base["Valore"] = stats_base["Valore"].apply(lambda x: f"{x:,}")
print("Tabella 1.1 — Statistiche di base (corpus originale)")
display(stats_base)
```

---

## AGGIUNTA 2 — Analisi duplicati (inserire prima della cella 20 di filtraggio)

Il report dedica una sezione (1.3.2) alla deduplicazione con conteggi before/after.
Aggiungi:

```python
# Sezione 1.3.2 — Analisi dei duplicati
n_prima = len(df)
dup_mask = df.duplicated(subset=["user_id", "item_id"], keep=False)
n_duplicati = dup_mask.sum()

# Deduplicazione: tieni l'ultima recensione per coppia (user, item)
df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
n_dopo = len(df)

tab_dup = pd.DataFrame({
    "Parametro": ["Recensioni iniziali", "Recensioni dopo deduplicazione", "Duplicati rimossi"],
    "Valore": [f"{n_prima:,}", f"{n_dopo:,}", f"{n_prima - n_dopo:,}"]
})
print(f"Duplicati trovati (coppie ripetute): {n_duplicati:,} ({n_duplicati/n_prima*100:.2f}% del totale)")
print("\nTabella 1.2 — Statistiche deduplicazione")
display(tab_dup)
```

---

## AGGIUNTA 3 — Sparsità e visualizzazione matrice ORIGINALE (inserire subito dopo l'aggiunta 2, prima del filtraggio)

Il report (1.3.3) mostra la sparsità PRIMA del filtraggio con una heatmap campione.

```python
# Sezione 1.3.3 — Sparsità della matrice originale
n_u_orig = df["user_id"].nunique()
n_i_orig = df["item_id"].nunique()
sparsita_orig = 1 - len(df) / (n_u_orig * n_i_orig)
print(f"Sparsità matrice originale: {sparsita_orig:.4%}  (prossima al 100%)")

# Visualizzazione: campione dei 100 utenti e 100 film più attivi
top_u_viz = df.groupby("user_id").size().sort_values(ascending=False).head(100).index
top_i_viz = df.groupby("item_id").size().sort_values(ascending=False).head(100).index
sample = df[df["user_id"].isin(top_u_viz) & df["item_id"].isin(top_i_viz)]
mat_viz = sample.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")
mat_viz = mat_viz.reindex(index=top_u_viz, columns=top_i_viz)

plt.figure(figsize=(8, 6))
plt.imshow(mat_viz.notna(), aspect="auto", cmap="Blues", interpolation="nearest")
plt.title("Matrice utente–prodotto originale (campione 100×100)")
plt.xlabel("Prodotti"); plt.ylabel("Utenti")
plt.colorbar(label="Rating presente")
plt.tight_layout(); plt.show()
print("Nota: la quasi totalità delle celle è vuota → sparsità prossima al 100%.")
```

---

## AGGIUNTA 4 — Tabella riepilogo + visualizzazione matrice FILTRATA (inserire subito dopo la cella 20 di filtraggio)

Il report (1.3.5) presenta una tabella formale e una heatmap della matrice filtrata.

```python
# Sezione 1.3.5 — Tabella riepilogativa matrice filtrata
n_u = df_cf["user_id"].nunique()
n_i = df_cf["item_id"].nunique()
n_r = len(df_cf)
dim = n_u * n_i
densita = n_r / dim

tab_filtrata = pd.DataFrame({
    "Parametro": ["Utenti", "Prodotti", "Rating osservati", "Dimensione matrice", "Densità", "Sparsità"],
    "Valore": [f"{n_u:,}", f"{n_i:,}", f"{n_r:,}", f"{dim:,}", f"{densita:.4%}", f"{1-densita:.4%}"]
})
print("Tabella 1.6 — Statistiche matrice utente–prodotto dopo il filtraggio")
display(tab_filtrata)

# Visualizzazione matrice filtrata (campione 100×100) per confronto con l'originale
top_u_f = df_cf.groupby("user_id").size().sort_values(ascending=False).head(100).index
top_i_f = df_cf.groupby("item_id").size().sort_values(ascending=False).head(100).index
samp_f = df_cf[df_cf["user_id"].isin(top_u_f) & df_cf["item_id"].isin(top_i_f)]
mat_f = samp_f.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")
mat_f = mat_f.reindex(index=top_u_f, columns=top_i_f)

plt.figure(figsize=(8, 6))
plt.imshow(mat_f.notna(), aspect="auto", cmap="Blues", interpolation="nearest")
plt.title("Matrice utente–prodotto filtrata (campione 100×100)")
plt.xlabel("Prodotti"); plt.ylabel("Utenti")
plt.colorbar(label="Rating presente")
plt.tight_layout(); plt.show()
```

---

## AGGIUNTA 5 — Sezione 1.4 completa: Analisi del dataset Metadata

Questa sezione manca del tutto nella Parte 1 del notebook (i metadata vengono
caricati solo nella Parte 2). Va aggiunta in fondo alla Parte 1, PRIMA delle
"Conclusioni Progetto Base" (cella 34).

### 5a — Caricamento metadata

```python
# === SEZIONE 1.4 — ANALISI DATASET METADATA ===
def load_meta(category, streaming=True, max_rows=None):
    url = f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_{category}.jsonl"
    ds = load_dataset("json", data_files=url, split="train", streaming=streaming)
    rows = []
    for ex in ds:
        rows.append(ex)
        if max_rows and len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)

meta = load_meta(CATEGORY)
print("Metadata shape:", meta.shape)
print("Campi disponibili:", list(meta.columns))
```

### 5b — Qualità dei dati (% valori mancanti)

```python
# Sezione 1.4.2 — Qualità dei dati: percentuale valori mancanti per campo
def pct_missing(col):
    s = meta[col]
    if s.apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        # campo lista: vuoto = lista vuota
        return s.apply(lambda x: len(x) == 0 if isinstance(x, (list, np.ndarray)) else pd.isna(x)).mean()
    return s.isna().mean()

quality = pd.DataFrame({
    "Campo": meta.columns,
    "% Mancanti": [f"{pct_missing(c)*100:.2f}%" for c in meta.columns]
})
print("Tabella 1.x — Qualità dei dati (valori mancanti per campo)")
display(quality.sort_values("% Mancanti", ascending=False))
```

### 5c — Analisi componenti testuali (titoli e descrizioni)

```python
# Sezione 1.4.3 — Analisi componenti testuali
def join_list(x):
    if isinstance(x, (list, np.ndarray)):
        return " ".join(map(str, x))
    return "" if x is None else str(x)

meta["title_str"] = meta["title"].fillna("").astype(str)
meta["desc_str"] = meta["description"].apply(join_list)

meta["title_len"] = meta["title_str"].str.split().apply(len)
meta["desc_len"] = meta["desc_str"].str.split().apply(len)

pct_desc_vuote = (meta["desc_len"] == 0).mean()

print("Titoli:")
print(f"  lunghezza media: {meta['title_len'].mean():.1f} parole | mediana: {meta['title_len'].median():.0f}")
print("\nDescrizioni:")
print(f"  lunghezza media: {meta['desc_len'].mean():.1f} parole | mediana: {meta['desc_len'].median():.0f}")
print(f"  descrizioni vuote: {pct_desc_vuote:.2%}")

# Grafici distribuzione
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(meta["title_len"], bins=40, color="steelblue")
axes[0].set_title("Distribuzione lunghezza titoli")
axes[0].set_xlabel("Numero parole"); axes[0].set_ylabel("Frequenza")
axes[1].hist(meta["desc_len"], bins=40, color="indianred")
axes[1].set_title("Distribuzione lunghezza descrizioni")
axes[1].set_xlabel("Numero parole"); axes[1].set_ylabel("Frequenza")
plt.tight_layout(); plt.show()
```

### 5d — Riduzione al subset e statistiche finali

```python
# Sezione 1.4.4 — Riduzione ai prodotti presenti nelle review filtrate
item_filtrati = set(df_cf["item_id"])
n_meta_orig = len(meta)
metadata_subset = meta[meta["parent_asin"].isin(item_filtrati)].copy()

print(f"Prodotti nel metadata originale: {n_meta_orig:,}")
print(f"Prodotti nel metadata_subset (intersezione con review): {len(metadata_subset):,}")
print(f"Prodotti persi: {n_meta_orig - len(metadata_subset):,}")

print("\nStatistiche testuali del subset:")
print(f"  lunghezza media titoli: {metadata_subset['title_len'].mean():.1f} parole")
print(f"  descrizioni vuote: {(metadata_subset['desc_len']==0).mean():.2%}")
print(f"  lunghezza media descrizioni: {metadata_subset['desc_len'].mean():.1f} parole")

# Scatter titolo vs descrizione (come Figura 1.3c del report)
plt.figure(figsize=(7, 5))
plt.scatter(metadata_subset["title_len"], metadata_subset["desc_len"], alpha=0.4, color="indianred")
plt.title("Titolo vs Descrizione (lunghezza in parole)")
plt.xlabel("Lunghezza titolo"); plt.ylabel("Lunghezza descrizione")
plt.tight_layout(); plt.show()
```

---

## Commit finale

```bash
git add "Progetto_MIGA_Completo_Movies (2).ipynb"
git commit -m "feat(cap1): completamento esplorazione e pulizia allineato al report

- Tabella statistiche base corpus originale
- Sezione deduplicazione con conteggi before/after
- Sparsità + visualizzazione matrice originale (campione 100x100)
- Tabella riepilogo + visualizzazione matrice filtrata
- Sezione 1.4 Metadata: caricamento, qualità dati, analisi testuale, riduzione al subset"
git push origin main
```

---

## Nota su differenza dal report di riferimento

Il report di Davide usa categoria Amazon Fashion con soglie 3/5; il nostro progetto
usa Movies_and_TV con soglie asimmetriche 5/10 e top-K solo utenti. I numeri assoluti
saranno diversi, ma la STRUTTURA delle analisi e delle tabelle è la stessa. Mantieni
le numerazioni delle tabelle (1.1, 1.2, 1.6, ecc.) coerenti col report per facilitare
la stesura.