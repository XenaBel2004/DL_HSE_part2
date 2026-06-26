import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


REACTIONS = [
    "like",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
]


class BaselineMLP(nn.Module):
    def __init__(self, n_users, item_dim=64, user_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, user_dim)
        self.net = nn.Sequential(
            nn.Linear(user_dim + item_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user, item_emb):
        u = self.user_emb(user)
        x = torch.cat([u, item_emb], dim=1)
        return self.net(x).squeeze(1)


class GeometryMLP(nn.Module):
    def __init__(self, n_users, item_dim=64, geom_dim=25, user_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, user_dim)

        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(user_dim + item_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user, item_emb, geometry):
        u = self.user_emb(user)
        g = self.geom_encoder(geometry)
        x = torch.cat([u, item_emb, g], dim=1)
        return self.net(x).squeeze(1)


@st.cache_data
def load_data():
    df = pd.read_parquet("data/vk_working_subset.parquet")

    matrices = np.load(
        "data/user_reaction_matrices.npy",
        allow_pickle=True,
    ).item()

    results = pd.read_csv("data/mlp_results.csv")

    user_classes = np.load(
        "data/user_encoder_classes.npy",
        allow_pickle=True,
    )

    return df, matrices, results, user_classes


@st.cache_resource
def load_models(n_users):
    baseline = BaselineMLP(n_users=n_users)
    geometry = GeometryMLP(n_users=n_users)

    baseline.load_state_dict(
        torch.load("data/baseline_mlp.pt", map_location="cpu")
    )

    geometry.load_state_dict(
        torch.load("data/geometry_mlp.pt", map_location="cpu")
    )

    baseline.eval()
    geometry.eval()

    return baseline, geometry


def predict_baseline(model, user_idx, item_emb):
    user = torch.tensor([user_idx], dtype=torch.long)
    item = torch.tensor([item_emb], dtype=torch.float32)

    with torch.no_grad():
        logit = model(user, item)
        prob = torch.sigmoid(logit).item()

    return prob


def predict_geometry(model, user_idx, item_emb, D):
    user = torch.tensor([user_idx], dtype=torch.long)
    item = torch.tensor([item_emb], dtype=torch.float32)
    geom = torch.tensor([D.flatten()], dtype=torch.float32)

    with torch.no_grad():
        logit = model(user, item, geom)
        prob = torch.sigmoid(logit).item()

    return prob


def plot_matrix(D, title):
    fig, ax = plt.subplots()

    im = ax.imshow(D)

    ax.set_xticks(range(len(REACTIONS)))
    ax.set_yticks(range(len(REACTIONS)))
    ax.set_xticklabels(REACTIONS, rotation=45, ha="right")
    ax.set_yticklabels(REACTIONS)

    for i in range(len(REACTIONS)):
        for j in range(len(REACTIONS)):
            ax.text(
                j,
                i,
                f"{D[i, j]:.2f}",
                ha="center",
                va="center",
            )

    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    return fig


def nearest_users(user_idx, matrices, top_k=5):
    D = matrices[user_idx]

    rows = []

    for other_idx, D_other in matrices.items():
        if other_idx == user_idx:
            continue

        dist = np.linalg.norm(D - D_other)

        rows.append({
            "user_idx": other_idx,
            "distance": dist,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("distance").head(top_k)

    return result


st.set_page_config(
    page_title="Reaction Geometry Recommender",
    layout="wide",
)

st.title("Reaction Geometry Recommender")

df, matrices, results, user_classes = load_data()

user_encoder = LabelEncoder()
user_encoder.classes_ = user_classes

df["user_idx"] = user_encoder.transform(df["user_id"])

baseline_model, geometry_model = load_models(
    n_users=len(user_classes)
)

tab1, tab2 = st.tabs([
    "Prediction Explorer",
    "Similar Users",
])



with tab1:
    st.header("Prediction Explorer")

    st.write(
        """
        Здесь можно выбрать пользователя и объект.
        Приложение сравнивает вероятность лайка по обычной модели
        и по модели с реакционной геометрией.
        """
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_user_id = st.selectbox(
            "User ID",
            df["user_id"].drop_duplicates().sort_values().tolist(),
        )

        user_idx = int(
            user_encoder.transform([selected_user_id])[0]
        )

        user_df = df[df["user_id"] == selected_user_id]

        selected_item_id = st.selectbox(
            "Item ID",
            user_df["item_id"].tolist(),
        )

        row = user_df[user_df["item_id"] == selected_item_id].iloc[0]

        item_emb = row["embedding"].astype("float32")

        if user_idx in matrices:
            D = matrices[user_idx]
        else:
            D = np.zeros((5, 5), dtype="float32")

        p_base = predict_baseline(
            baseline_model,
            user_idx,
            item_emb,
        )

        p_geom = predict_geometry(
            geometry_model,
            user_idx,
            item_emb,
            D,
        )

        delta = p_geom - p_base

        st.metric("Baseline P(like)", f"{p_base:.4f}")
        st.metric("Geometry P(like)", f"{p_geom:.4f}")
        st.metric("Difference", f"{delta:+.4f}")

        st.write("True like:")
        st.code(bool(row["like"]))

    with col_right:
        st.subheader("User reaction geometry")

        D_df = pd.DataFrame(
            D,
            index=REACTIONS,
            columns=REACTIONS,
        )

        st.dataframe(D_df)

        fig = plot_matrix(
            D,
            title=f"Reaction geometry for user {selected_user_id}",
        )

        st.pyplot(fig)

    st.subheader("Interpretation")

    closest = []
    farthest = []

    for i in range(len(REACTIONS)):
        for j in range(i + 1, len(REACTIONS)):
            closest.append((REACTIONS[i], REACTIONS[j], D[i, j]))
            farthest.append((REACTIONS[i], REACTIONS[j], D[i, j]))

    closest = sorted(closest, key=lambda x: x[2])[:3]
    farthest = sorted(farthest, key=lambda x: -x[2])[:3]

    c1, c2 = st.columns(2)

    with c1:
        st.write("Самые похожие реакции:")
        for a, b, d in closest:
            st.write(f"- {a} ↔ {b}: {d:.4f}")

    with c2:
        st.write("Самые разные реакции:")
        for a, b, d in farthest:
            st.write(f"- {a} ↔ {b}: {d:.4f}")

    st.subheader("Model results")

    st.dataframe(results)


with tab2:
    st.header("Similar Users")

    st.write(
        """
        Здесь можно найти пользователей с похожей реакционной геометрией.
        Расстояние считается как Frobenius distance между матрицами D_u.
        """
    )

    selected_user_id_2 = st.selectbox(
        "Choose user",
        df["user_id"].drop_duplicates().sort_values().tolist(),
        key="similar_user_select",
    )

    user_idx_2 = int(
        user_encoder.transform([selected_user_id_2])[0]
    )

    top_k = st.slider(
        "Number of similar users",
        min_value=3,
        max_value=20,
        value=5,
    )

    if user_idx_2 not in matrices:
        st.warning("Для этого пользователя нет рассчитанной геометрии.")
    else:
        similar = nearest_users(
            user_idx_2,
            matrices,
            top_k=top_k,
        )

        similar["user_id"] = user_encoder.inverse_transform(
            similar["user_idx"].astype(int)
        )

        st.subheader("Nearest users")

        st.dataframe(
            similar[["user_id", "user_idx", "distance"]]
        )

        st.subheader("Selected user geometry")

        D_main = matrices[user_idx_2]

        fig_main = plot_matrix(
            D_main,
            title=f"Selected user {selected_user_id_2}",
        )

        st.pyplot(fig_main)

        st.subheader("Compare with nearest user")

        nearest_idx = int(similar.iloc[0]["user_idx"])
        nearest_id = similar.iloc[0]["user_id"]

        D_nearest = matrices[nearest_idx]

        c1, c2 = st.columns(2)

        with c1:
            st.write(f"Selected user: {selected_user_id_2}")
            st.pyplot(
                plot_matrix(
                    D_main,
                    title="Selected user",
                )
            )

        with c2:
            st.write(f"Nearest user: {nearest_id}")
            st.pyplot(
                plot_matrix(
                    D_nearest,
                    title="Nearest user",
                )
            )

        st.write(
            "Distance:",
            float(np.linalg.norm(D_main - D_nearest)),
        )