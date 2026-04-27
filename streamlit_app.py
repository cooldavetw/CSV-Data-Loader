import io
import os
import re
from typing import List

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------
# DB CONFIG
# ---------------------------------------------------------------------
PG_HOST = os.getenv("PG_HOST", "pgvector-db")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "sEgMa6")
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")
DEFAULT_SCHEMA = os.getenv("PG_SCHEMA", "public")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def sanitize_identifier(name: str, label: str) -> str:
    """
    Allow letters, digits, underscore, spaces, and CJK characters.
    Disallow quotes/semicolons to avoid SQL injection; return stripped name.
    """
    if not name:
        raise ValueError(f"{label} cannot be empty")

    cleaned = name.strip()
    if not cleaned:
        raise ValueError(f"{label} cannot be empty")
    if '"' in cleaned or ";" in cleaned or "." in cleaned:
        raise ValueError(f"{label} cannot contain quotes, semicolons, or dots")

    allowed_pattern = r"^[A-Za-z0-9_\s\u4e00-\u9fff]+$"
    if not re.match(allowed_pattern, cleaned):
        raise ValueError(
            f"{label} may contain letters, digits, underscore, spaces, and Chinese characters only"
        )
    return cleaned


def quote_identifier(name: str, label: str = "Identifier") -> str:
    """Return a safely double-quoted SQL identifier."""
    return f'"{sanitize_identifier(name, label)}"'


def normalize_column_name(name: object, position: int) -> str:
    """Make CSV column names valid, stable, and readable PostgreSQL identifiers."""
    raw = "" if name is None else str(name)
    cleaned = raw.strip().replace("\x00", "")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or f"column_{position}"


def normalize_columns(columns: List[object]) -> List[str]:
    normalized = []
    seen = {}

    for idx, column in enumerate(columns, start=1):
        base = normalize_column_name(column, idx)
        count = seen.get(base, 0)
        seen[base] = count + 1
        normalized.append(base if count == 0 else f"{base}_{count + 1}")

    return normalized


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    driver_candidates = ["psycopg2", "psycopg"]
    last_error = None

    for driver in driver_candidates:
        url = (
            f"postgresql+{driver}://{PG_USER}:{PG_PASSWORD}"
            f"@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
        )
        try:
            engine = create_engine(url, future=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "No PostgreSQL driver available. Install `psycopg2-binary` (preferred) or `psycopg`."
    ) from last_error


def list_tables(engine: Engine, schema: str) -> List[str]:
    schema = sanitize_identifier(schema, "Schema name")
    inspector = inspect(engine)
    return sorted(inspector.get_table_names(schema=schema))


def ensure_schema(engine: Engine, schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema, 'Schema name')}"))


@st.cache_data(show_spinner=False)
def parse_csv(
    file_bytes: bytes,
    encoding: str,
    separator: str,
    header_row: bool,
) -> pd.DataFrame:
    header = 0 if header_row else None
    sep = separator or ","
    df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=sep, header=header)

    if header_row:
        df.columns = normalize_columns(list(df.columns))
    else:
        df.columns = [f"column_{idx}" for idx in range(1, len(df.columns) + 1)]

    return df


def load_csv_to_postgres(
    engine: Engine,
    df: pd.DataFrame,
    schema: str,
    table_name: str,
    if_exists: str,
) -> None:
    schema = sanitize_identifier(schema, "Schema name")
    table_name = sanitize_identifier(table_name, "Table name")

    ensure_schema(engine, schema)
    df.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )


# ---------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------
def main():
    st.title("CSV資料庫載入器")

    st.sidebar.header("PostgreSQL 設定")
    st.sidebar.caption(f"{PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}")

    engine = get_engine()

    st.subheader("1. 選擇目標資料表")
    schema = st.text_input("Schema", value=DEFAULT_SCHEMA)

    try:
        existing_tables = list_tables(engine, schema)
    except Exception as exc:
        st.error(f"Cannot list tables for schema `{schema}`: {exc}")
        return

    table_options = ["<Create new table>"] + existing_tables
    table_choice = st.selectbox("Target table", table_options)

    new_table_name = ""
    if table_choice == "<Create new table>":
        new_table_name = st.text_input("New table name")

    if_exists_label = st.radio(
        "When the table already exists",
        ["Append rows", "Replace table", "Fail with an error"],
        horizontal=True,
    )
    if_exists_map = {
        "Append rows": "append",
        "Replace table": "replace",
        "Fail with an error": "fail",
    }

    st.subheader("2. 上傳CSV文件")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    col1, col2, col3 = st.columns(3)
    with col1:
        encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "big5", "latin1"])
    with col2:
        separator = st.text_input("Separator", value=",", max_chars=4)
    with col3:
        header_row = st.checkbox("First row has headers", value=True)

    df = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        try:
            df = parse_csv(file_bytes, encoding, separator, header_row)
        except Exception as exc:
            st.error(f"Cannot read CSV file: {exc}")
            return

    st.subheader("3. 預覽資料")
    if df is None:
        st.info("Upload a CSV to preview rows before loading.")
    elif df.empty:
        st.warning("The CSV file was parsed successfully, but it contains no rows.")
    else:
        st.write(f"{len(df):,} rows x {len(df.columns):,} columns")
        st.dataframe(df.head(100), use_container_width=True)

    st.subheader("4. 載入PostgreSQL")
    if st.button("Load CSV into PostgreSQL"):
        if df is None or df.empty:
            st.error("No CSV rows to load.")
            return

        if table_choice == "<Create new table>":
            target_table = new_table_name.strip()
        else:
            target_table = table_choice

        try:
            target_schema = sanitize_identifier(schema, "Schema name")
            target_table = sanitize_identifier(target_table, "Table name")
        except ValueError as exc:
            st.error(f"Invalid target: {exc}")
            return

        with st.spinner("Loading CSV into PostgreSQL ..."):
            try:
                load_csv_to_postgres(
                    engine=engine,
                    df=df,
                    schema=target_schema,
                    table_name=target_table,
                    if_exists=if_exists_map[if_exists_label],
                )
            except Exception as exc:
                st.error(f"Database load failed: {exc}")
                return

        st.success(
            f'Loaded {len(df):,} rows into {quote_identifier(target_schema, "Schema name")}.'
            f'{quote_identifier(target_table, "Table name")}.'
        )


if __name__ == "__main__":
    main()
