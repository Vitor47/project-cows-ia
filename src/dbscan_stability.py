from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


def dbscan_stability(X_scaled, eps_values=[1.0, 1.5, 2.0], min_samples=5):
    """
    Mede a estabilidade do DBSCAN comparando diferentes valores de eps.
    Retorna a média dos ARI (Adjusted Rand Index).
    """
    clusterings = []

    # Rodar DBSCAN com diferentes eps
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        clusterings.append(db.labels_)

    # Comparar cada clusterização com a de referência (primeira)
    reference = clusterings[0]
    ari_scores = [adjusted_rand_score(reference, c) for c in clusterings[1:]]

    return ari_scores, (
        sum(ari_scores) / len(ari_scores) if ari_scores else None
    )
