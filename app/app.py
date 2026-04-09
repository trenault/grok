import html
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATABASE_PATH = PROJECT_DIR / "annotatr.db"
DOCUMENTS_TABLE = "documents"
ANNOTATIONS_TABLE = "annotations"
ANNOTATORS_TABLE = "annotators"
DEFAULT_ANNOTATORS = ["Dave", "Thomas", "Mohsen"]

st.set_page_config(page_title="AnnotaTR", layout="wide")


def ensure_annotators_table():
    with get_connection() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {ANNOTATORS_TABLE} (
                name TEXT PRIMARY KEY
            )
        """)
        for name in DEFAULT_ANNOTATORS:
            conn.execute(
                f"INSERT OR IGNORE INTO {ANNOTATORS_TABLE} (name) VALUES (?)",
                (name,),
            )
        conn.commit()


def load_annotators():
    ensure_annotators_table()
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT name FROM {ANNOTATORS_TABLE} ORDER BY rowid"
        ).fetchall()
    return [row[0] for row in rows]


def add_annotator(name):
    ensure_annotators_table()
    with get_connection() as conn:
        conn.execute(
            f"INSERT OR IGNORE INTO {ANNOTATORS_TABLE} (name) VALUES (?)",
            (name.strip(),),
        )
        conn.commit()


def select_annotator():
    if st.session_state.get("annotator"):
        return True

    st.title("AnnotaTR")
    st.subheader("Who are you?")

    annotators = load_annotators()
    cols = st.columns(len(annotators))
    for i, name in enumerate(annotators):
        if cols[i].button(name, use_container_width=True, key=f"annotator_{name}"):
            st.session_state["annotator"] = name
            st.rerun()

    st.divider()
    with st.form("add_annotator_form", clear_on_submit=True):
        new_name = st.text_input("Add a new annotator", placeholder="Enter name…")
        submitted = st.form_submit_button("Add", use_container_width=False)

    if submitted:
        new_name = new_name.strip()
        if not new_name:
            st.error("Please enter a name.")
        elif new_name in annotators:
            st.warning(f"**{new_name}** is already in the list.")
        else:
            add_annotator(new_name)
            st.rerun()

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


def ensure_annotations_table():
    with get_connection() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {ANNOTATIONS_TABLE} (
                m_id TEXT NOT NULL,
                annotator TEXT NOT NULL,
                label TEXT NOT NULL,
                PRIMARY KEY (m_id, annotator)
            )
        """)
        conn.commit()


def import_documents(uploaded_file):
    incoming = pd.read_csv(uploaded_file)
    if "m_id" not in incoming.columns:
        raise ValueError("The uploaded CSV must include an 'm_id' column.")

    incoming = incoming.copy()
    incoming["m_id"] = incoming["m_id"].astype(str)
    incoming = incoming.drop_duplicates(subset="m_id", keep="first").reset_index(drop=True)

    for col in ("label", "label_dave", "label_thomas", "label_mohsen"):
        if col in incoming.columns:
            incoming = incoming.drop(columns=[col])

    existing = load_documents()
    if existing.empty:
        incoming.insert(0, "source_order", range(1, len(incoming) + 1))
        with get_connection() as conn:
            incoming.to_sql(DOCUMENTS_TABLE, conn, if_exists="replace", index=False)
        return len(incoming)

    existing_ids = set(existing["m_id"].astype(str))
    new_docs = incoming[~incoming["m_id"].isin(existing_ids)].copy()
    if new_docs.empty:
        return 0

    next_source_order = int(existing["source_order"].max()) + 1
    new_docs.insert(0, "source_order", range(next_source_order, next_source_order + len(new_docs)))

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


def save_label(doc_id, label, annotator):
    ensure_annotations_table()
    with get_connection() as conn:
        conn.execute(
            f"""INSERT INTO {ANNOTATIONS_TABLE} (m_id, annotator, label)
                VALUES (?, ?, ?)
                ON CONFLICT(m_id, annotator) DO UPDATE SET label = excluded.label""",
            (str(doc_id), annotator, label),
        )
        conn.commit()


def get_annotator_labeled_ids(annotator):
    ensure_annotations_table()
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT m_id FROM {ANNOTATIONS_TABLE} WHERE annotator = ?",
            (annotator,),
        ).fetchall()
    return {row[0] for row in rows}


def get_next_doc(documents, annotator):
    if documents.empty:
        return None, None

    labeled_ids = get_annotator_labeled_ids(annotator)
    unlabeled = documents[~documents["m_id"].isin(labeled_ids)]

    if unlabeled.empty:
        return None, None

    doc = unlabeled.iloc[0]
    return doc["m_id"], doc


def get_annotator_stats(documents, annotator):
    if documents.empty:
        return 0, 0, 0.0

    total = len(documents)
    done = len(get_annotator_labeled_ids(annotator))
    remaining = total - done
    progress = done / total if total > 0 else 0.0
    return remaining, done, progress


def get_labeled_export(annotator):
    ensure_annotations_table()
    documents = load_documents()
    if documents.empty:
        return None

    with get_connection() as conn:
        annotations_df = pd.read_sql_query(
            f"SELECT m_id, label FROM {ANNOTATIONS_TABLE} WHERE annotator = ?",
            conn,
            params=(annotator,),
        )

    if annotations_df.empty:
        return None

    labeled = documents.merge(annotations_df, on="m_id", how="inner")
    return labeled.to_csv(index=False).encode("utf-8")


def get_all_annotations():
    ensure_annotations_table()
    with get_connection() as conn:
        df = pd.read_sql_query(
            f"SELECT m_id, annotator, label FROM {ANNOTATIONS_TABLE} ORDER BY m_id, annotator",
            conn,
        )
    return df


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


@st.dialog("All Annotations", width="large")
def show_all_annotations():
    df = get_all_annotations()
    if df.empty:
        st.info("No annotations have been saved yet.")
        return

    pivot = df.pivot_table(index="m_id", columns="annotator", values="label", aggfunc="first")
    pivot = pivot.reset_index()
    pivot.columns.name = None
    st.dataframe(pivot, use_container_width=True)
    st.caption(f"Total annotations: {len(df)}")


if not select_annotator():
    st.stop()

annotator = st.session_state["annotator"]

st.sidebar.title("AnnotaTR")
st.sidebar.caption(f"Annotating as: **{annotator}**")

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
remaining, done, progress_value = get_annotator_stats(documents, annotator)
total = len(documents)

st.sidebar.metric("Remaining", remaining)
st.sidebar.progress(progress_value)
st.sidebar.caption(f"{done} of {total} documents classified")

export_bytes = get_labeled_export(annotator)
st.sidebar.download_button(
    "Download labeled CSV",
    data=export_bytes or b"",
    file_name="labeled_documents.csv",
    mime="text/csv",
    disabled=export_bytes is None,
    use_container_width=True,
)

if st.sidebar.button("View all annotations", use_container_width=True):
    show_all_annotations()

if st.sidebar.button("Log out", use_container_width=True):
    del st.session_state["annotator"]
    st.rerun()

if st.sidebar.button("Reload data", use_container_width=True):
    st.rerun()

st.title("Document Classification")

if documents.empty:
    st.info("Upload a CSV file with an 'm_id' column to start labeling.")
    st.stop()

doc_id, doc = get_next_doc(documents, annotator)

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
        save_label(doc_id, key, annotator)
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

