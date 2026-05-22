

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # ← this line fixes blank charts in Streamlit
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

st.set_page_config(
    page_title="Student AI Dashboard",
    page_icon="🎓",
    layout="wide",
)

@st.cache_resource
def train_models():
    np.random.seed(42)

    data = {
        'attendance_pct':   np.concatenate([
            np.random.normal(88, 6,  40),
            np.random.normal(65, 8,  40),
            np.random.normal(38, 10, 40),
        ]),
        'cgpa':             np.concatenate([
            np.random.normal(8.2, 0.5, 40),
            np.random.normal(6.5, 0.7, 40),
            np.random.normal(4.8, 0.8, 40),
        ]),
        'assignments_done': np.concatenate([
            np.random.normal(9.0, 0.8, 40),
            np.random.normal(6.5, 1.2, 40),
            np.random.normal(3.5, 1.5, 40),
        ]),
        'backlogs':         np.concatenate([
            np.random.normal(0.2, 0.4, 40),
            np.random.normal(1.8, 0.8, 40),
            np.random.normal(4.5, 1.2, 40),
        ]),
        'participation':    np.concatenate([
            np.random.normal(8.5, 1.0, 40),
            np.random.normal(5.5, 1.2, 40),
            np.random.normal(2.5, 1.0, 40),
        ]),
    }

    df = pd.DataFrame(data)

    # Define features FIRST
    features = ['attendance_pct', 'cgpa', 'assignments_done',
                'backlogs', 'participation']

    # Clip to valid ranges
    df['attendance_pct']   = df['attendance_pct'].clip(0, 100)
    df['cgpa']             = df['cgpa'].clip(0, 10)
    df['assignments_done'] = df['assignments_done'].clip(0, 10)
    df['backlogs']         = df['backlogs'].clip(0, 10)
    df['participation']    = df['participation'].clip(0, 10)

    # Dropout label
    df['dropout_risk'] = (df['cgpa'] < 6.0).astype(int)
    df.loc[(df['attendance_pct'] < 50) & (df['backlogs'] > 3), 'dropout_risk'] = 1

    # Random Forest
    X = df[features]
    y = df['dropout_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))

    # Scaler
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + Silhouette
    inertias   = []
    sil_scores = []
    K_range    = list(range(2, 11))
    for k in K_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_temp.fit(X_scaled)
        inertias.append(km_temp.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km_temp.labels_))

    # Final K-Means
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X_scaled)
    df['cluster'] = km.labels_

    # Name clusters
    summary         = df.groupby('cluster')['cgpa'].mean()
    sorted_clusters = summary.sort_values(ascending=False).index.tolist()
    cluster_names   = {
        sorted_clusters[0]: 'High Performers',
        sorted_clusters[1]: 'Average Students',
        sorted_clusters[2]: 'At-Risk Students',
    }

    # PCA
    pca                = PCA(n_components=2, random_state=42)
    X_2d               = pca.fit_transform(X_scaled)
    df['pca1']         = X_2d[:, 0]
    df['pca2']         = X_2d[:, 1]
    variance_explained = pca.explained_variance_ratio_ * 100

    return (rf, km, pca, scaler,
            cluster_names, acc, df, features,
            inertias, sil_scores, K_range,
            variance_explained)


(rf_model, km_model, pca_model, scaler,
 cluster_names, model_acc, df, features,
 inertias, sil_scores, K_range,
 variance_explained) = train_models()

COLORS = {0: '#1D9E75', 1: '#D85A30', 2: '#7F77DD'}
COLORS_LIST = ['#1D9E75', '#D85A30', '#7F77DD']

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🎓 SGSITS Student AI Dashboard")
st.caption("Random Forest · K-Means Clustering · PCA · Silhouette Analysis")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Accuracy",       f"{model_acc*100:.1f}%")
c2.metric("Training Dataset",     f"{len(df)} students")
c3.metric("At-Risk in Dataset",   f"{df['dropout_risk'].mean()*100:.0f}%")
c4.metric("Best Silhouette Score",f"{max(sil_scores):.3f}")

st.divider()

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Predict Student",
    "Cluster Analysis",
    "PCA Visualization",
    "Elbow & Silhouette",
    "Model Insights",
])


with tab1:
    st.subheader("Enter Student Details")
    st.caption("Move sliders — predictions update instantly.")

    left, right = st.columns(2)

    with left:
        attendance    = st.slider("Attendance %",     0,   100,  75)
        cgpa          = st.slider("CGPA",             0.0, 10.0, 7.0, step=0.1)
        assignments   = st.slider("Assignments Done", 0,   10,   7)
        backlogs      = st.slider("Active Backlogs",  0,   10,   1)
        participation = st.slider("Participation",    0,   10,   6)

    student_input  = np.array([[attendance, cgpa, assignments,
                                backlogs, participation]])
    dropout_pred   = rf_model.predict(student_input)[0]
    dropout_prob   = rf_model.predict_proba(student_input)[0]
    student_scaled = scaler.transform(student_input)
    cluster_id     = km_model.predict(student_scaled)[0]
    cluster_label  = cluster_names[cluster_id]

    with right:
        st.markdown("### Dropout Prediction")

        if dropout_pred == 1:
            st.error("🔴 **DROPOUT RISK DETECTED**")
        else:
            st.success("🟢 **STUDENT IS ON TRACK**")

        st.markdown(f"**Risk probability:** {dropout_prob[1]*100:.1f}%")
        st.progress(int(dropout_prob[1] * 100))
        st.markdown(f"**Safe probability:** {dropout_prob[0]*100:.1f}%")
        st.progress(int(dropout_prob[0] * 100))

        st.divider()

        emoji_map = {
            'High Performers'
            'Average Students'
            'At-Risk Students'
        }
        st.markdown("### Cluster Group")
        st.info(f"{emoji_map.get(cluster_label)} **{cluster_label}**")

        st.markdown("### Recommended Action")
        if dropout_pred == 1:
            st.warning("""
            - Schedule counsellor meeting this week
            - Review backlog clearance plan
            - Contact parents if attendance < 50%
            - Assign peer mentor
            """)
        elif cluster_label == 'Average Students':
            st.info("""
            - Monitor monthly
            - Encourage project participation
            - Consider skill development workshops
            """)
        else:
            st.success("""
            - Student performing well
            - Consider for mentorship roles
            - Recommend higher-level challenges
            """)


with tab2:
    st.subheader("Student Cluster Profiles")
    st.caption("K-Means (K=3) finds natural student groups — no labels used.")

    cluster_feature_means = (
        df.groupby('cluster')[features].mean()
          .rename(index=cluster_names)
    )

    left2, right2 = st.columns(2)

    with left2:
        cluster_counts = df['cluster'].value_counts().sort_index()
        labels_named   = [cluster_names[i] for i in cluster_counts.index]

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.pie(
            cluster_counts.values,
            labels=labels_named,
            autopct='%1.0f%%',
            colors=COLORS_LIST,
            startangle=90,
            textprops={'fontsize': 11}
        )
        ax1.set_title('Student Distribution Across Clusters')
        st.pyplot(fig1)
        plt.close(fig1)

    with right2:
        st.markdown("**Average values per cluster:**")
        st.dataframe(cluster_feature_means.round(2), use_container_width=True)
        sil_k3 = sil_scores[1]
        st.metric("Silhouette Score at K=3", f"{sil_k3:.3f}")

    st.divider()
    st.markdown("### Feature Comparison Across Clusters")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    sorted_c = df.groupby('cluster')['cgpa'].mean() \
                 .sort_values(ascending=False).index

    for idx, c in enumerate(sorted_c):
        ax  = axes2[idx]
        col = COLORS_LIST[idx]
        vals = cluster_feature_means.loc[cluster_names[c]]
        norm = (vals - df[features].min()) / \
               (df[features].max() - df[features].min())

        bars = ax.barh(features, norm.values, color=col, alpha=0.85)
        ax.set_xlim(0, 1.15)
        ax.set_title(cluster_names[c], fontweight='bold', color=col)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.2, axis='x')

        for bar, raw_val in zip(bars, vals.values):
            ax.text(bar.get_width() + 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f'{raw_val:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


with tab3:
    st.subheader("PCA — 5D Data Reduced to 2D")
    st.caption(
        f"First 2 components explain "
        f"{variance_explained[0]:.1f}% + {variance_explained[1]:.1f}% = "
        f"{variance_explained.sum():.1f}% of total variance."
    )

    # Scatter plot
    fig3, ax3 = plt.subplots(figsize=(10, 7))

    for c in range(3):
        mask = df['cluster'] == c
        ax3.scatter(
            df[mask]['pca1'], df[mask]['pca2'],
            c=COLORS[c], label=cluster_names[c],
            alpha=0.7, s=80,
            edgecolors='white', linewidth=0.5
        )

    centroids_2d = pca_model.transform(km_model.cluster_centers_)
    for c in range(3):
        ax3.scatter(
            centroids_2d[c, 0], centroids_2d[c, 1],
            marker='*', s=400, c=COLORS[c],
            edgecolors='black', linewidth=1.5, zorder=5
        )
        ax3.annotate(
            f'  {cluster_names[c]}\n  centroid',
            (centroids_2d[c, 0], centroids_2d[c, 1]),
            fontsize=9, color=COLORS[c], fontweight='bold'
        )

    ax3.set_xlabel(f'PCA Component 1 ({variance_explained[0]:.1f}% variance)')
    ax3.set_ylabel(f'PCA Component 2 ({variance_explained[1]:.1f}% variance)')
    ax3.set_title('Student Clusters — Visualized in 2D via PCA')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Variance bar chart
    st.markdown("### How Much Does Each Component Capture?")
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    bars_var = ax4.bar(
        ['Component 1', 'Component 2'],
        variance_explained,
        color=['#1D9E75', '#3B8BD4'],
        edgecolor='white'
    )
    for bar, val in zip(bars_var, variance_explained):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )
    ax4.set_ylabel('Variance Explained (%)')
    ax4.set_title(f'Total variance preserved: {variance_explained.sum():.1f}%')
    ax4.set_ylim(0, max(variance_explained) + 12)
    ax4.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)


with tab4:
    st.subheader("Finding the Right Number of Clusters")
    st.caption(
        "Elbow Method: look for where inertia stops dropping sharply. "
        "Silhouette Score: higher = better separated clusters. "
        "Both should agree on the same K."
    )

    fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    fig5.suptitle(
        'How Many Clusters? — Elbow Method + Silhouette Score',
        fontsize=13, fontweight='bold'
    )

    # Elbow
    ax5.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax5.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Best K = 3')
    ax5.set_xlabel('Number of Clusters (K)')
    ax5.set_ylabel('Inertia (lower = tighter clusters)')
    ax5.set_title('Elbow Method')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.annotate(
        'Elbow\nK=3',
        xy=(3, inertias[1]),
        xytext=(4.5, inertias[1] + 20),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=10, color='red'
    )

    # Silhouette
    ax6.plot(K_range, sil_scores, 'go-', linewidth=2, markersize=8)
    ax6.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Best K = 3')
    ax6.set_xlabel('Number of Clusters (K)')
    ax6.set_ylabel('Silhouette Score (higher = better)')
    ax6.set_title('Silhouette Score')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    best_k_idx = sil_scores.index(max(sil_scores))
    ax6.annotate(
        f'Peak\nK={K_range[best_k_idx]}',
        xy=(K_range[best_k_idx], max(sil_scores)),
        xytext=(K_range[best_k_idx] + 1.5, max(sil_scores) - 0.03),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=10, color='red'
    )

    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # Scores table
    st.markdown("### Full Scores Table")
    scores_df = pd.DataFrame({
        'K (clusters)':     K_range,
        'Inertia':          [round(v, 1) for v in inertias],
        'Silhouette Score': [round(v, 3) for v in sil_scores],
    })
    st.dataframe(scores_df, use_container_width=True, hide_index=True)

    st.info("""
    **How to read this:**
    - Inertia always drops as K increases — so don't just pick the lowest.
    - Look for where the **drop slows down** — that is the elbow.
    - Silhouette Score tells you cluster quality. Pick the **peak**.
    - Both pointing to K=3 gives you confidence in the choice.
    """)


with tab5:
    st.subheader("Random Forest — Feature Importance")
    st.caption("Which student metrics matter most for dropout prediction?")

    importances       = rf_model.feature_importances_
    indices           = np.argsort(importances)[::-1]
    sorted_features   = [features[i] for i in indices]
    sorted_importance = importances[indices]

    fig6, ax7 = plt.subplots(figsize=(8, 4))
    bar_colors = ['#1D9E75' if i == 0 else '#3B8BD4'
                  for i in range(len(features))]
    bars6 = ax7.bar(
        sorted_features, sorted_importance * 100,
        color=bar_colors, edgecolor='white', linewidth=0.5
    )
    for bar, val in zip(bars6, sorted_importance * 100):
        ax7.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )
    ax7.set_ylabel('Importance (%)')
    ax7.set_title('Feature Importance — Random Forest (100 trees)')
    ax7.set_ylim(0, max(sorted_importance * 100) + 8)
    ax7.tick_params(axis='x', rotation=10)
    ax7.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

    st.divider()
    st.subheader("Model Performance Report")

    report = classification_report(
        df['dropout_risk'],
        rf_model.predict(df[features]),
        target_names=['Safe', 'At Risk'],
        output_dict=True
    )
    report_df = pd.DataFrame(report).T.drop('accuracy', errors='ignore')
    st.dataframe(report_df.round(3), use_container_width=True)

    st.info("""
    **How to read this:**
    - **Precision** — of students flagged at-risk, what % actually are?
    - **Recall** — of all at-risk students, what % did we catch?
    - **F1-score** — balance of both. Higher = better.

    For dropout prediction, **recall matters more** — missing an at-risk
    student is worse than a false alarm.
    """)