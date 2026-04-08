import streamlit as st
import pandas as pd
import os
import html
import hmac
from pathlib import Path
from streamlit.errors import StreamlitSecretNotFoundError

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "to_annotate.csv"
LABELS_PATH = PROJECT_DIR / "labeled" / "labels.csv"

st.set_page_config(page_title="AnnotaTR", layout="wide")

LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["m_id"] = df["m_id"].astype(str)
    return df.drop_duplicates(subset="m_id", keep="first")


def load_labels():
    if os.path.exists(LABELS_PATH):
        labels = pd.read_csv(LABELS_PATH, index_col=0)
        labels.index = labels.index.astype(str)
        labels.index.name = "m_id"
        return labels[~labels.index.duplicated(keep="last")]
    return pd.DataFrame(columns=["label"], index=pd.Index([], name="m_id"))


def save_label(doc, label):
    labels = load_labels()
    doc_id = str(doc["m_id"])
    row_to_save = doc.copy()
    row_to_save["m_id"] = doc_id
    row_to_save["label"] = label

    for column in row_to_save.index:
        if column not in labels.columns:
            labels[column] = pd.NA

    labels.loc[doc_id, row_to_save.index.tolist()] = row_to_save.values
    labels.to_csv(LABELS_PATH)


def get_auth_credentials():
    try:
        username = st.secrets.get("app_username", "test1")
        password = st.secrets.get("app_password", "test2")
    except StreamlitSecretNotFoundError:
        username = "test1"
        password = "test2"
    return str(username), str(password)


def login_required():
    expected_username, expected_password = get_auth_credentials()

    if st.session_state.get("authenticated", False):
        return True

    st.title("AnnotaTR Login")
    st.caption("Sign in to access the annotation app.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        valid_username = hmac.compare_digest(username, expected_username)
        valid_password = hmac.compare_digest(password, expected_password)
        if valid_username and valid_password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Invalid username or password.")

    return False


def get_unlabeled_docs(df, labels):
    labeled_ids = set(labels.index.tolist())
    return df[~df["m_id"].isin(labeled_ids)]


def get_next_doc(unlabeled):
    if unlabeled.empty:
        return None, None
    doc = unlabeled.iloc[0]
    doc_id = doc["m_id"]
    return doc_id, doc


def render_text_block(kind, text):
    colors = {
        "info": ("#eaf4ff", "#0f172a", "#90caf9"),
        "warning": ("#fff6e8", "#0f172a", "#f6c177"),
        "success": ("#ecfdf3", "#0f172a", "#7ad39b"),
    }
    background, foreground, border = colors[kind]
    safe_text = html.escape("" if pd.isna(text) else str(text))
    st.markdown(
        f"""
        <div style="
            background:{background};
            color:{foreground};
            border:1px solid {border};
            border-radius:0.5rem;
            padding:1rem;
            white-space:pre-wrap;
            overflow-wrap:anywhere;
            line-height:1.5;
        ">{safe_text}</div>
        """,
        unsafe_allow_html=True,
    )


if not login_required():
    st.stop()


# --- Load data ---
df = load_data()
labels = load_labels()
unlabeled = get_unlabeled_docs(df, labels)

total = len(df)
remaining = len(unlabeled)
done = total - remaining
progress_value = min(max(done / total, 0.0), 1.0) if total > 0 else 0.0

# --- Sidebar progress ---
st.sidebar.title("AnnotaTR")
st.sidebar.metric("Remaining", remaining)
st.sidebar.progress(progress_value)
st.sidebar.caption(f"{done} of {total} documents classified")
if st.sidebar.button("Log out"):
    st.session_state["authenticated"] = False
    st.rerun()

if st.sidebar.button("Reload data"):
    st.cache_data.clear()
    st.rerun()

# --- Main area ---
st.title("Document Classification")

doc_id, doc = get_next_doc(unlabeled)

if doc is None:
    st.success("All documents have been classified!")
    st.balloons()
else:

    st.divider()
    st.subheader("Your classification")

    cols = st.columns(4)
    labels_map = {
        "True": ("True", "green"),
        "False": ("False", "red"),
        "Uncertain": ("Uncertain", "orange"),
        "Invalid": ("Invalid", "gray"),
    }

    for i, (label, (display, _)) in enumerate(labels_map.items()):
        if cols[i].button(display, key=label, use_container_width=True):
            save_label(doc, label)
            st.rerun()
            
    st.caption(f"Document ID: `{doc_id}`")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tweet")
        render_text_block("info", doc.get("t_text", ""))

        st.subheader("Mention / Query")
        render_text_block("warning", doc.get("m_text", ""))

    with col2:
        st.subheader("LLM Response")
        render_text_block("success", doc.get("l_text", ""))

        score = doc.get("l_score", None)
        if pd.notna(score):
            st.metric("LLM Score", f"{score:.0f} / 100")

    
