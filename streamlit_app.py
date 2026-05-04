import io
import os
import re
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning


# ---------------------------------------------------------------------
# DB CONFIG
# ---------------------------------------------------------------------
PG_HOST = os.getenv("PG_HOST", "pgvector-db")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "sEgMa6")
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")
DEFAULT_SCHEMA = os.getenv("PG_SCHEMA", "public")

SEGMA_API_BASE_URL = os.getenv("SEGMA_API_BASE_URL", "http://backend:3040")
SEGMA_DATA_SOURCES_PATH = os.getenv("SEGMA_DATA_SOURCES_PATH", "/api/v1/data_sources")
SEGMA_ACTION_DATASETS_PATH = os.getenv(
    "SEGMA_ACTION_DATASETS_PATH", "/api/v1/action_datasets"
)

LLM_API_KEY = os.getenv("LLM_API_KEY", "abcd")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm-proxy:4000/v1")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))

class GeneratedCsv(BaseModel):
    csv_text: str = Field(min_length=1)
    separator: str = Field(default=",", min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_csv(self) -> "GeneratedCsv":
        try:
            df = pd.read_csv(io.StringIO(self.csv_text), sep=self.separator, header=0)
        except Exception as exc:
            raise ValueError(f"Generated text is not valid CSV: {exc}") from exc

        if df.empty:
            raise ValueError("Generated CSV must contain at least one data row")
        if len(df.columns) == 0:
            raise ValueError("Generated CSV must contain at least one column")
        if any(str(column).strip() == "" for column in df.columns):
            raise ValueError("Generated CSV cannot contain blank column names")

        normalized = normalize_columns(list(df.columns))
        if len(set(normalized)) != len(normalized):
            raise ValueError("Generated CSV column names must be unique after normalization")

        return self


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


def qualified_table_name(schema: str, table_name: str) -> str:
    return (
        f'{quote_identifier(schema, "Schema name")}.'
        f'{quote_identifier(table_name, "Table name")}'
    )


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
        conn.execute(
            text(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema, 'Schema name')}")
        )


def build_api_url(base_url: str, path: str) -> str:
    if not base_url.strip():
        raise ValueError("Segma API base URL is required")
    if not path.strip():
        raise ValueError("Segma API path is required")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def configure_tls_warnings() -> None:
    disable_warnings(InsecureRequestWarning)


def segma_headers(api_token: str, include_content_type: bool = False) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
    }
    if include_content_type:
        headers["Content-Type"] = "application/json"
    if api_token.strip():
        headers["Authorization"] = f"Bearer {api_token.strip()}"
    return headers


def extract_segma_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = (
            payload.get("data_sources")
            or payload.get("datasources")
            or payload.get("dataSources")
            or payload.get("items")
            or payload.get("results")
            or payload.get("data")
            or []
        )
    else:
        items = []

    return [item for item in items if isinstance(item, dict)]


def datasource_id(datasource: Dict[str, Any]) -> Any:
    for key in ("id", "data_source_id", "datasource_id", "uuid"):
        value = datasource.get(key)
        if value is not None and value != "":
            return value
    raise ValueError("Selected Segma data source does not include an id")


def datasource_label(datasource: Dict[str, Any]) -> str:
    name = (
        datasource.get("name")
        or datasource.get("title")
        or datasource.get("display_name")
        or datasource.get("displayName")
        or datasource_id(datasource)
    )
    return str(name)


def fetch_segma_data_sources(
    base_url: str,
    api_token: str,
    data_sources_path: str,
) -> List[Dict[str, Any]]:
    configure_tls_warnings()
    response = requests.get(
        build_api_url(base_url, data_sources_path),
        headers=segma_headers(api_token),
        verify=False,
        timeout=30,
    )
    response.raise_for_status()
    return extract_segma_items(response.json())


def create_segma_action_dataset(
    base_url: str,
    api_token: str,
    action_datasets_path: str,
    data_source: Dict[str, Any],
    dataset_name: str,
    sql: str,
) -> Dict[str, Any]:
    configure_tls_warnings()
    payload = {
        "name": dataset_name,
        "data_source_id": datasource_id(data_source),
        "model_type": "sql",
        "sql": sql,
    }
    response = requests.post(
        build_api_url(base_url, action_datasets_path),
        headers=segma_headers(api_token, include_content_type=True),
        json=payload,
        verify=False,
        timeout=30,
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return {"status_code": response.status_code, "text": response.text}


def strip_markdown_fence(text_value: str) -> str:
    cleaned = text_value.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def generate_csv_with_llm(
    api_key: str,
    base_url: str,
    model: str,
    user_prompt: str,
    row_count: int,
    separator: str,
    timeout_seconds: int,
) -> str:
    if not api_key.strip():
        raise ValueError("LLM API key is required")
    if not base_url.strip():
        raise ValueError("LLM base URL is required")
    if not model.strip():
        raise ValueError("LLM model name is required")
    if not user_prompt.strip():
        raise ValueError("Describe the CSV data you want to generate")

    system_prompt = (
        "You generate clean RFC 4180-style CSV data. "
        "Return only CSV text with one header row and data rows. "
        "Do not include Markdown fences, prose, comments, or explanations."
    )
    prompt = (
        f"Generate {row_count} rows of CSV data using `{separator}` as the delimiter. "
        "Column names must be non-empty and unique. "
        "Every row must have the same number of fields as the header. "
        f"Data requirements: {user_prompt.strip()}"
    )
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise TimeoutError(
            f"LLM request timed out after {timeout_seconds} seconds. "
            "Try a smaller row count, a faster model, or a higher timeout."
        ) from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "error"
        response_text = exc.response.text[:500] if exc.response is not None else ""
        message = f"LLM API returned HTTP {status_code}"
        if response_text:
            message = f"{message}: {response_text}"
        raise RuntimeError(message) from exc
    except requests.RequestException as exc:
        raise ConnectionError(f"Cannot reach LLM API at {base_url.strip()}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("LLM response was not valid JSON") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response did not include chat completion content") from exc

    return strip_markdown_fence(content)


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


def parse_generated_csv(csv_text: str, separator: str) -> pd.DataFrame:
    GeneratedCsv(csv_text=csv_text, separator=separator)
    df = pd.read_csv(io.StringIO(csv_text), sep=separator, header=0)
    df.columns = normalize_columns(list(df.columns))
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
    st.sidebar.header("LLM 設定")
    llm_api_key = st.sidebar.text_input(
        "LLM API key",
        type="password",
        help="API key for completions.",
        value=LLM_API_KEY,
    )
    llm_base_url = st.sidebar.text_input(
        "LLM base URL",
        value=LLM_BASE_URL,
        help="Base URL for OpenAI-compatible LLM endpoint.",
    )
    llm_model = st.sidebar.text_input(
        "LLM model name",
        value=LLM_MODEL,
        help="Model name for answering questions.",
    )
    llm_timeout_seconds = st.sidebar.number_input(
        "LLM timeout seconds",
        min_value=5,
        max_value=900,
        value=LLM_TIMEOUT_SECONDS,
        step=5,
        help="Maximum time to wait for CSV generation.",
    )
    st.sidebar.header("Segma API 設定")
    segma_base_url = st.sidebar.text_input("Segma API Base URL", value=SEGMA_API_BASE_URL)

    if "token" not in st.session_state:
        token_from_url = st.query_params.get("token", None)
        if token_from_url:
            st.session_state.token = token_from_url

    segma_api_token = st.sidebar.text_input(
        "API bearer token",
        value=st.session_state.get("token", ""),
        type="password",
    )
    st.session_state.token = segma_api_token

    segma_data_sources_path = st.sidebar.text_input(
        "Data sources path",
        value=SEGMA_DATA_SOURCES_PATH,
    )
    segma_action_datasets_path = st.sidebar.text_input(
        "Action datasets path",
        value=SEGMA_ACTION_DATASETS_PATH,
    )

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

    st.subheader("2. 選擇資料來源")
    source_mode = st.radio(
        "Data source",
        ["Upload CSV file", "Generate CSV with LLM"],
        horizontal=True,
    )
    df = None

    if source_mode == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        col1, col2, col3 = st.columns(3)
        with col1:
            encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "big5", "latin1"])
        with col2:
            separator = st.text_input("Separator", value=",", max_chars=4)
        with col3:
            header_row = st.checkbox("First row has headers", value=True)

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            try:
                df = parse_csv(file_bytes, encoding, separator, header_row)
            except Exception as exc:
                st.error(f"Cannot read CSV file: {exc}")
                return
    else:
        llm_prompt = st.text_area(
            "Describe the CSV data to generate",
            value="Create sample customer data with customer_id, name, email, city, signup_date, and total_spend.",
            height=120,
        )
        col1, col2 = st.columns(2)
        with col1:
            generated_rows = st.number_input(
                "Rows to generate",
                min_value=1,
                max_value=500,
                value=25,
                step=1,
            )
        with col2:
            generated_separator = st.text_input(
                "Generated CSV separator",
                value=",",
                max_chars=4,
            )

        if st.button("Generate CSV data"):
            with st.spinner("Generating CSV data with LLM ..."):
                try:
                    generated_csv = generate_csv_with_llm(
                        llm_api_key,
                        llm_base_url,
                        llm_model,
                        llm_prompt,
                        int(generated_rows),
                        generated_separator,
                        int(llm_timeout_seconds),
                    )
                    parse_generated_csv(generated_csv, generated_separator)
                except TimeoutError as exc:
                    st.error(str(exc))
                except (ConnectionError, RuntimeError) as exc:
                    st.error(f"LLM CSV generation failed: {exc}")
                except Exception as exc:
                    st.error(f"LLM CSV generation failed validation: {exc}")
                else:
                    st.session_state.generated_csv_text = generated_csv
                    st.success("Generated CSV passed validation.")

        generated_csv_text = st.text_area(
            "Generated CSV",
            value=st.session_state.get("generated_csv_text", ""),
            height=220,
        )
        if generated_csv_text:
            try:
                df = parse_generated_csv(generated_csv_text, generated_separator)
            except Exception as exc:
                st.error(f"Generated CSV is not valid for PostgreSQL loading: {exc}")
                return

    st.subheader("3. 預覽資料")
    if df is None:
        st.info("Upload or generate CSV data to preview rows before loading.")
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
            f"Loaded {len(df):,} rows into {qualified_table_name(target_schema, target_table)}."
        )
        st.session_state.last_loaded_table = {
            "schema": target_schema,
            "table": target_table,
            "rows": len(df),
        }

    st.subheader("5. 建立Segma action_dataset")
    last_loaded = st.session_state.get("last_loaded_table")
    if last_loaded:
        target_schema = last_loaded["schema"]
        target_table = last_loaded["table"]
        st.caption(
            f"Last loaded table: {qualified_table_name(target_schema, target_table)} "
            f"({last_loaded['rows']:,} rows)"
        )
    else:
        st.info("Load a CSV into PostgreSQL first, then create a Segma action_dataset.")

    if st.button("Refresh Segma data sources"):
        try:
            st.session_state.segma_data_sources = fetch_segma_data_sources(
                segma_base_url,
                segma_api_token,
                segma_data_sources_path,
            )
        except Exception as exc:
            st.error(f"Cannot fetch Segma data sources: {exc}")

    segma_data_sources = st.session_state.get("segma_data_sources", [])
    if segma_data_sources:
        data_source = st.selectbox(
            "Segma data source",
            segma_data_sources,
            format_func=datasource_label,
        )
    else:
        data_source = None
        st.caption("No Segma data sources loaded yet.")

    dataset_name_default = ""
    action_sql = ""
    if last_loaded:
        dataset_name_default = f"{last_loaded['table']}_action_dataset"
        action_sql = f"SELECT * FROM {qualified_table_name(last_loaded['schema'], last_loaded['table'])}"

    dataset_name = st.text_input("action_dataset name", value=dataset_name_default)
    action_sql = st.text_area("SQL", value=action_sql, height=140)

    if st.button("Create Segma action_dataset"):
        if not last_loaded:
            st.error("Load a CSV into PostgreSQL before creating the action_dataset.")
            return
        if data_source is None:
            st.error("Select a Segma data source first.")
            return
        if not dataset_name.strip():
            st.error("action_dataset name is required.")
            return
        if not action_sql.strip():
            st.error("SQL is required.")
            return

        with st.spinner("Creating Segma action_dataset ..."):
            try:
                result = create_segma_action_dataset(
                    segma_base_url,
                    segma_api_token,
                    segma_action_datasets_path,
                    data_source,
                    dataset_name.strip(),
                    action_sql.strip(),
                )
            except Exception as exc:
                st.error(f"Segma action_dataset creation failed: {exc}")
                return

        st.success("Segma action_dataset created.")
        st.json(result)


if __name__ == "__main__":
    main()
