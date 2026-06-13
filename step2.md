Contesto e obiettivo

Allineare la Parte 1 del notebook (Collaborative Filtering) alla struttura del
Capitolo 2 del report di riferimento. Il notebook ha già: filtraggio dati (df_cf
pronto), una grid search base con KNNWithMeans. Va REIMPOSTATA la sezione CF per
seguire esattamente le sottosezioni 2.2 → 2.8 del report.

Punto di partenza assunto: esiste già df_cf (dataset filtrato con colonne
user_id, item_id, rating, timestamp) prodotto dal Capitolo 1.

REGOLE:


Ogni cella di codice deve essere preceduta da una cella Markdown che spiega
in italiano cosa fa e perché (sono le spiegazioni qui sotto in blockquote).
Mantieni RANDOM_STATE = 42 per riproducibilità.
Usa la libreria Surprise dove possibile. Usa KNNBaseline (non KNNWithMeans)
per avere i bias bu/bi via ALS, coerentemente col report.
Sostituisci la vecchia grid search; non lasciare codice duplicato.



SEZIONE 2.2 — Preparazione della Matrice Utente–Prodotto

Cella Markdown (spiegazione)


2.2 Preparazione della matrice utente–prodotto. Costruiamo la matrice
sparsa utente×item in cui ogni cella contiene il rating (o resta vuota se
l'interazione non esiste). Prima dividiamo i dati in train (80%) e test (20%):
il mapping da ID originali a indici interi viene definito SOLO sul train, così
da simulare correttamente uno scenario reale ed evitare il cold-start (utenti
o item presenti nel test ma mai visti nel train vengono scartati). Usiamo il
formato sparso CSR per efficienza di memoria.



Cella codice

pythonimport numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# 2.2.2 Train-test split (80/20)
train_df, test_df = train_test_split(
    df_cf[["user_id", "item_id", "rating"]],
    test_size=0.20, random_state=RANDOM_STATE
)

# 2.2.3 Indicizzazione: mapping ID -> interi consecutivi, definito SUL TRAIN
user_ids = train_df["user_id"].unique()
item_ids = train_df["item_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {it: i for i, it in enumerate(item_ids)}

# Rimuovi dal test gli utenti/item non visti nel train (cold-start)
n_test_before = len(test_df)
test_df = test_df[test_df["user_id"].isin(user2idx) & test_df["item_id"].isin(item2idx)]
print(f"Indicizzazione: utenti = 0..{len(user_ids)-1}, item = 0..{len(item_ids)-1}")
print(f"Righe test rimosse per cold-start: {n_test_before - len(test_df)}")

# 2.2.4 Costruzione matrice utente-prodotto (COO -> CSR)
rows = train_df["user_id"].map(user2idx).values
cols = train_df["item_id"].map(item2idx).values
vals = train_df["rating"].values
M = coo_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids))).tocsr()
print(f"Matrice utente-prodotto: {M.shape}, non-zero: {M.nnz}")


SEZIONE 2.3 — Ricerca della Configurazione Ottimale del Modello K-NN

Cella Markdown (spiegazione)


2.3 Configurazione ottimale K-NN. Usiamo KNNBaseline di Surprise, che
calcola le componenti di baseline (media globale µ, bias utente bu, bias item
bi) tramite ALS, e poi affina la predizione con i K vicini più simili pesati
dalla similarità. Eseguiamo una grid search sulle combinazioni: similarità
(cosine, pearson), shrinkage (10, 50, 100), modalità (user-based / item-based),
e valori di K (5,10,20,30,40,60,80,100). Per ogni configurazione calcoliamo
RMSE e MSE sul test set, e selezioniamo quella con errore minimo.



Cella codice

pythonfrom surprise import Dataset, Reader, KNNBaseline, accuracy
from surprise.model_selection import PredefinedKFold
import itertools

reader = Reader(rating_scale=(1, 5))

# Ricostruiamo train/test in formato Surprise mantenendo lo split già fatto
trainset = Dataset.load_from_df(train_df[["user_id","item_id","rating"]], reader).build_full_trainset()
testset = list(test_df[["user_id","item_id","rating"]].itertuples(index=False, name=None))

# Griglia parametri (come nel report)
sim_names = ["cosine", "pearson"]
shrinkages = [10, 50, 100]
user_based_opts = [True, False]
k_values = [5, 10, 20, 30, 40, 60, 80, 100]

results = []
for sim_name, shrink, ub in itertools.product(sim_names, shrinkages, user_based_opts):
    for k in k_values:
        algo = KNNBaseline(
            k=k,
            sim_options={"name": sim_name, "user_based": ub, "shrinkage": shrink},
            bsl_options={"method": "als"},
            verbose=False
        )
        algo.fit(trainset)
        preds = algo.test(testset)
        rmse = accuracy.rmse(preds, verbose=False)
        mse = rmse ** 2
        results.append({
            "similarita": sim_name, "shrinkage": shrink,
            "modalita": "user-based" if ub else "item-based",
            "k": k, "RMSE": rmse, "MSE": mse
        })

res_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
print("Top 10 configurazioni per RMSE:")
display(res_df.head(10))

Cella Markdown (spiegazione)


2.3.3 Risultati sperimentali. Visualizziamo l'andamento dell'RMSE al variare
di K, separato per ciascuna combinazione di similarità e modalità, per capire
dove si trova il minimo e come si comporta il modello.



Cella codice

pythonimport matplotlib.pyplot as plt

best = res_df.iloc[0]
print("=== CONFIGURAZIONE OTTIMALE ===")
print(f"Similarità: {best['similarita']} | Shrinkage: {best['shrinkage']} | "
      f"Modalità: {best['modalita']} | k: {best['k']}")
print(f"RMSE = {best['RMSE']:.5f} | MSE = {best['MSE']:.5f}")

# Grafico RMSE vs k per la configurazione vincente (fissando similarità+shrinkage+modalità)
mask = ((res_df["similarita"]==best["similarita"]) &
        (res_df["shrinkage"]==best["shrinkage"]) &
        (res_df["modalita"]==best["modalita"]))
sub = res_df[mask].sort_values("k")

plt.figure(figsize=(8,5))
plt.plot(sub["k"], sub["RMSE"], marker="o", label="RMSE")
plt.plot(sub["k"], sub["MSE"], marker="s", label="MSE")
plt.axvline(best["k"], color="red", linestyle="--", label=f"Best k={best['k']}")
plt.xlabel("k (numero vicini)"); plt.ylabel("Valore errore")
plt.title(f"RMSE e MSE al variare di k ({best['similarita']}, {best['modalita']}, shrink={best['shrinkage']})")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


SEZIONE 2.4 — Filling della Matrice di Rating

Cella Markdown (spiegazione)


2.4 Filling della matrice. Usando la configurazione ottimale, riaddestriamo
il modello su tutto il dataset e prediciamo TUTTI i rating mancanti (le coppie
utente-item non osservate). Otteniamo così una matrice densa, dove ogni cella
ha un valore (osservato o predetto), pronta per il clustering e le raccomandazioni.



Cella codice

python# Riaddestra il modello ottimale sull'intero df_cf
full_data = Dataset.load_from_df(df_cf[["user_id","item_id","rating"]], reader)
full_trainset = full_data.build_full_trainset()

best_algo = KNNBaseline(
    k=int(best["k"]),
    sim_options={"name": best["similarita"],
                 "user_based": best["modalita"]=="user-based",
                 "shrinkage": int(best["shrinkage"])},
    bsl_options={"method": "als"},
    verbose=False
)
best_algo.fit(full_trainset)

# Predici le coppie non osservate (anti-testset)
anti = full_trainset.build_anti_testset()
print(f"Coppie mancanti da predire: {len(anti):,}")
pred_missing = best_algo.test(anti)

# Costruisci la matrice densa: osservati + predetti
all_users = df_cf["user_id"].unique()
all_items = df_cf["item_id"].unique()
rating_matrix = pd.DataFrame(index=all_users, columns=all_items, dtype=float)

# riempi con gli osservati
for r in df_cf.itertuples(index=False):
    rating_matrix.at[r.user_id, r.item_id] = r.rating
# riempi con i predetti
for p in pred_missing:
    rating_matrix.at[p.uid, p.iid] = p.est

print(f"Matrice densa: {rating_matrix.shape}, valori predetti inseriti: {len(pred_missing):,}")


SEZIONE 2.5 — Segmentazione degli Utenti (Clustering)

Cella Markdown (spiegazione)


2.5 Segmentazione utenti. Ogni utente è ora un vettore di rating (una riga
della matrice densa). Riduciamo la dimensionalità con PCA (20 componenti per il
clustering, 2 per la visualizzazione). Poi applichiamo K-Means: per usare la
cosine similarity normalizziamo i vettori a norma L2 (su vettori normalizzati
la distanza euclidea è equivalente alla cosine). Scegliamo il numero di cluster
k con Elbow Method (inertia) e Silhouette Score.



Cella codice

pythonfrom sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 2.5.1 Preprocessing + PCA
X = rating_matrix.fillna(rating_matrix.mean()).values
X_norm = normalize(X, norm="l2")   # per cosine via euclidea

pca20 = PCA(n_components=20, random_state=RANDOM_STATE)
X_pca = pca20.fit_transform(X_norm)

# 2.5.2 Scelta di k: Elbow + Silhouette
K_range = range(2, 15)
inertias, silhouettes = [], []
for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))

fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(list(K_range), inertias, marker="o", color="#4A90D9")
ax[0].set_title("Elbow Method - KMeans"); ax[0].set_xlabel("Numero di cluster (k)"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(K_range), silhouettes, marker="o", color="#7AC142")
ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("Numero di cluster (k)"); ax[1].set_ylabel("Silhouette")
plt.tight_layout(); plt.show()

Cella Markdown (spiegazione)


2.5.3 Clustering finale. Scelto il k ottimale (in base ai due criteri sopra),
eseguiamo il K-Means definitivo e profiliamo ogni cluster: numero di utenti,
rating medio, recensioni medie per utente. Questo permette di interpretare i
segmenti (utenti positivi, equilibrati, critici).



Cella codice

python# Imposta best_k_clusters in base ai grafici sopra (es. 6); modificabile
best_k_clusters = 6

km = KMeans(n_clusters=best_k_clusters, init="k-means++", n_init=10, random_state=RANDOM_STATE)
user_clusters = pd.Series(km.fit_predict(X_pca), index=rating_matrix.index, name="cluster")

# Profilazione dei cluster
recensioni_per_user = df_cf.groupby("user_id").size()
prof = pd.DataFrame({
    "cluster": user_clusters,
    "rating_medio": rating_matrix.mean(axis=1),
    "recensioni": recensioni_per_user.reindex(rating_matrix.index)
})
summary = prof.groupby("cluster").agg(
    n_utenti=("rating_medio", "size"),
    rating_medio=("rating_medio", "mean"),
    recensioni_per_user=("recensioni", "mean")
).round(2)
print("Statistiche dei cluster individuati:")
display(summary)

Cella Markdown (spiegazione)


2.5.4 Visualizzazione cluster. Proiettiamo gli utenti nello spazio delle
prime 2 componenti principali, colorando per cluster, per vedere la separazione
dei segmenti.



Cella codice

pythonpca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca2.fit_transform(X_norm)

plt.figure(figsize=(9,6))
for c in range(best_k_clusters):
    mask = user_clusters.values == c
    plt.scatter(X_2d[mask,0], X_2d[mask,1], s=12, alpha=0.6, label=f"Cluster {c}")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Cluster Utenti (PCA 2D)")
plt.legend(); plt.tight_layout(); plt.show()


SEZIONE 2.6 — Generazione delle Raccomandazioni Top-N

Cella Markdown (spiegazione)


2.6 Raccomandazioni Top-N. Per ogni utente ordiniamo gli item NON ancora
valutati in base al rating predetto, e prendiamo i primi N. Mostriamo poi le
raccomandazioni per un utente rappresentativo di ciascun cluster, per osservare
come i segmenti ricevano suggerimenti coerenti col loro profilo.



Cella codice

pythonTOP_N = 10
rated_by_user = df_cf.groupby("user_id")["item_id"].agg(set).to_dict()

def top_n_for_user(user, n=TOP_N):
    seen = rated_by_user.get(user, set())
    row = rating_matrix.loc[user]
    candidates = row[~row.index.isin(seen)].sort_values(ascending=False)
    return candidates.head(n)

# Un utente rappresentativo per cluster (il primo di ciascun gruppo)
print("Top-N raccomandazioni per un utente rappresentativo di ogni cluster:\n")
for c in range(best_k_clusters):
    rep_user = user_clusters[user_clusters==c].index[0]
    recs = top_n_for_user(rep_user)
    print(f"--- Cluster {c} | utente {rep_user} | rating medio predetto: {recs.mean():.2f} ---")
    for item, est in recs.items():
        print(f"    {item}  ->  {est:.2f}")
    print()


SEZIONE 2.7 — Matrix Factorization (SVD) e Confronto Finale

Cella Markdown (spiegazione)


2.7 Matrix Factorization (SVD). Applichiamo un secondo approccio: la
decomposizione SVD, che approssima la matrice con k fattori latenti, catturando
strutture nascoste nelle preferenze. Confrontiamo SVD col miglior KNNBaseline
in termini di RMSE/MSE, usando lo stesso train/test split per un confronto equo.



Cella codice

pythonfrom surprise import SVD

# SVD addestrato sullo stesso trainset, valutato sullo stesso testset
svd = SVD(n_factors=50, random_state=RANDOM_STATE)
svd.fit(trainset)
svd_preds = svd.test(testset)
svd_rmse = accuracy.rmse(svd_preds, verbose=False)
svd_mse = svd_rmse ** 2

confronto = pd.DataFrame({
    "Modello": ["KNNBaseline (ottimizzato)", "SVD (50 fattori latenti)"],
    "RMSE": [best["RMSE"], svd_rmse],
    "MSE": [best["MSE"], svd_mse]
}).round(4)
print("Confronto KNNBaseline vs SVD:")
display(confronto)

ax = confronto.set_index("Modello")[["RMSE","MSE"]].plot(
    kind="bar", figsize=(7,4), color=["#7AC142","#4A90D9"], rot=0)
ax.set_ylabel("Errore"); ax.set_title("KNNBaseline vs SVD")
plt.tight_layout(); plt.show()

miglioramento = (best["RMSE"] - svd_rmse) / best["RMSE"] * 100
print(f"Variazione RMSE di SVD rispetto a KNNBaseline: {miglioramento:+.1f}%")

Cella Markdown (spiegazione)


2.7.3 Analisi delle raccomandazioni. Confrontiamo qualitativamente le Top-10
dei due modelli per un utente campione: misuriamo l'overlap (quanti item in
comune) e la dispersione dei punteggi predetti, per capire se SVD produce
raccomandazioni più diversificate rispetto al KNN.



Cella codice

python# Costruisci la matrice densa anche per SVD per confronto Top-N
sample_user = rating_matrix.index[0]

# Top-N da KNN (già in rating_matrix)
knn_top = set(top_n_for_user(sample_user).index)

# Top-N da SVD: predici gli item non visti per sample_user
seen = rated_by_user.get(sample_user, set())
svd_scores = {it: svd.predict(sample_user, it).est for it in all_items if it not in seen}
svd_top_series = pd.Series(svd_scores).sort_values(ascending=False).head(TOP_N)
svd_top = set(svd_top_series.index)

overlap = len(knn_top & svd_top)
print(f"Utente campione: {sample_user}")
print(f"Overlap Top-{TOP_N} tra KNN e SVD: {overlap}/{TOP_N} item in comune")
print(f"Dispersione punteggi KNN (std): {top_n_for_user(sample_user).std():.3f}")
print(f"Dispersione punteggi SVD (std): {svd_top_series.std():.3f}")


SEZIONE 2.8 — Conclusioni

Cella Markdown (spiegazione)


2.8 Conclusioni Collaborative Filtering. Riepilogo dei risultati: la
configurazione KNN ottimale trovata, il confronto con SVD, i punti di forza
(semplicità, interpretabilità) e i limiti del CF (sensibilità alla sparsità,
cold-start, assenza di informazione sul contenuto — che motiva il passaggio al
Content-Based del capitolo successivo). Riempire con i numeri ottenuti.



Cella codice

pythonprint("="*55)
print("RIEPILOGO COLLABORATIVE FILTERING")
print("="*55)
print(f"KNN ottimale: {best['similarita']}, {best['modalita']}, "
      f"shrink={best['shrinkage']}, k={best['k']}")
print(f"  RMSE = {best['RMSE']:.4f} | MSE = {best['MSE']:.4f}")
print(f"SVD (50 fattori): RMSE = {svd_rmse:.4f} | MSE = {svd_mse:.4f}")
print(f"Numero cluster utenti: {best_k_clusters}")
print(f"Modello migliore: {'SVD' if svd_rmse < best['RMSE'] else 'KNNBaseline'}")
print("="*55)


Commit finale

bashgit add "Progetto_MIGA_Completo_Movies (2).ipynb"
git commit -m "feat(cap2): collaborative filtering allineato al report

- 2.2 matrice utente-prodotto con train/test split e indicizzazione sul train
- 2.3 grid search KNNBaseline (cosine/pearson x shrinkage x user/item x k) con ALS
- 2.4 filling matrice densa con configurazione ottimale
- 2.5 segmentazione utenti K-Means (PCA + cosine L2, Elbow + Silhouette)
- 2.6 raccomandazioni Top-N per utente rappresentativo di cluster
- 2.7 Matrix Factorization SVD e confronto con KNN
- 2.8 conclusioni riepilogative"
git push origin main


Note


Se la grid search completa (96 configurazioni) risulta troppo lenta sui 3000
utenti × 12000 film, riduci temporaneamente k_values o usa cv ridotta, ma
documenta la scelta. Il collo di bottiglia è il calcolo delle similarità per
le combinazioni item-based.
best_k_clusters va impostato MANUALMENTE dopo aver guardato i grafici Elbow
e Silhouette: non lasciarlo fisso a 6 senza verificare che sia il valore giusto
per i tuoi dati.