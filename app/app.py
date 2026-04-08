import html
import hmac
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATABASE_PATH = PROJECT_DIR / "annotatr.db"
DOCUMENTS_TABLE = "documents"

st.set_page_config(page_title="AnnotaTR", layout="wide")


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


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def documents_table_exists():
    with get_connection() as conn:
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (DOCUMENTS_TABLE,),
        ).fetchone()
    return result is not None


def import_documents(uploaded_file):
    incoming = pd.read_csv(uploaded_file)
    if "m_id" not in incoming.columns:
        raise ValueError("The uploaded CSV must include an 'm_id' column.")

    incoming = incoming.copy()
    incoming["m_id"] = incoming["m_id"].astype(str)
    incoming = incoming.drop_duplicates(subset="m_id", keep="first").reset_index(drop=True)

    if "label" in incoming.columns:
        incoming = incoming.drop(columns=["label"])

    existing = load_documents()
    if existing.empty:
        incoming.insert(0, "source_order", range(1, len(incoming) + 1))
        incoming["label"] = pd.NA
        with get_connection() as conn:
            incoming.to_sql(DOCUMENTS_TABLE, conn, if_exists="replace", index=False)
        return len(incoming)

    existing_ids = set(existing["m_id"].astype(str))
    new_docs = incoming[~incoming["m_id"].isin(existing_ids)].copy()
    if new_docs.empty:
        return 0

    next_source_order = int(existing["source_order"].max()) + 1
    new_docs.insert(0, "source_order", range(next_source_order, next_source_order + len(new_docs)))
    new_docs["label"] = pd.NA

    all_columns = list(existing.columns)
    for column in new_docs.columns:
        if column not in all_columns:
            all_columns.append(column)
            existing[column] = pd.NA

    for column in all_columns:
        if column not in new_docs.columns:
            new_docs[column] = pd.NA

    combined = pd.concat(
        [existing[all_columns], new_docs[all_columns]],
        ignore_index=True,
    )

    with get_connection() as conn:
        combined.to_sql(DOCUMENTS_TABLE, conn, if_exists="replace", index=False)

    return len(new_docs)


def load_documents():
    if not documents_table_exists():
        return pd.DataFrame()

    with get_connection() as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {DOCUMENTS_TABLE} ORDER BY source_order",
            conn,
        )

    if "m_id" in df.columns:
        df["m_id"] = df["m_id"].astype(str)
    return df


def save_label(doc_id, label):
    with get_connection() as conn:
        conn.execute(
            f"UPDATE {DOCUMENTS_TABLE} SET label = ? WHERE m_id = ?",
            (label, str(doc_id)),
        )
        conn.commit()


def get_next_doc(documents):
    if documents.empty:
        return None, None

    unlabeled = documents[documents["label"].isna() | (documents["label"] == "")]
    if unlabeled.empty:
        return None, None

    doc = unlabeled.iloc[0]
    return doc["m_id"], doc


def get_labeled_export():
    documents = load_documents()
    if documents.empty:
        return None

    labeled = documents[documents["label"].notna() & (documents["label"] != "")]
    if labeled.empty:
        return None

    return labeled.to_csv(index=False).encode("utf-8")


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


st.sidebar.title("AnnotaTR")

with st.sidebar.form("upload_form"):
    uploaded_file = st.file_uploader("Upload documents CSV", type="csv")
    upload_submitted = st.form_submit_button("Load CSV", use_container_width=True)

if upload_submitted:
    if uploaded_file is None:
        st.sidebar.error("Choose a CSV file before loading.")
    else:
        try:
            inserted_count = import_documents(uploaded_file)
            if inserted_count == 0:
                st.sidebar.info("No new documents were added.")
            else:
                st.sidebar.success(f"Added {inserted_count} new documents to SQLite.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(str(exc))

documents = load_documents()

total = len(documents)
remaining = 0
done = 0
progress_value = 0.0

if total > 0:
    remaining = int((documents["label"].isna() | (documents["label"] == "")).sum())
    done = total - remaining
    progress_value = done / total

st.sidebar.metric("Remaining", remaining)
st.sidebar.progress(progress_value)
st.sidebar.caption(f"{done} of {total} documents classified")

export_bytes = get_labeled_export()
st.sidebar.download_button(
    "Download labeled CSV",
    data=export_bytes or b"",
    file_name="labeled_documents.csv",
    mime="text/csv",
    disabled=export_bytes is None,
    use_container_width=True,
)

if st.sidebar.button("Log out", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

if st.sidebar.button("Reload data", use_container_width=True):
    st.rerun()

st.title("Document Classification")

if documents.empty:
    st.info("Upload a CSV file with an 'm_id' column to start labeling.")
    st.stop()

doc_id, doc = get_next_doc(documents)

if doc is None:
    st.success("All documents have been classified!")
    st.balloons()
    st.stop()

st.divider()
st.subheader("Your classification")

cols = st.columns(4)
labels_map = {
    "True": "True",
    "False": "False",
    "Uncertain": "Uncertain",
    "Invalid": "Invalid",
}

for i, (key, display) in enumerate(labels_map.items()):
    if cols[i].button(display, key=key, use_container_width=True):
        save_label(doc_id, key)
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
