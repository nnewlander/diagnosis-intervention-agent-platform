from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration loaded from environment variables and .env."""

    PROJECT_NAME: str = "核桃智能教学诊断与干预 Agent 平台"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DATA_ROOT: str = "project1_agent_raw_data_10pct"
    DEBUG: bool = True
    TOP_K_RAG: int = 5
    TOP_K_KG: int = 5
    TOP_K_PACKAGES: int = 3
    MAX_SUBMISSIONS: int = 10
    OUTPUT_DIR: str = "outputs"
    RAG_PROVIDER: str = "local"
    RAG_API_BASE: str = "http://127.0.0.1:8001"
    RAG_ENDPOINT: str = "/search"
    RAG_API_KEY: str = ""
    RAG_TIMEOUT: int = 10
    RAG_RESPONSE_STYLE: str = "auto"
    KG_PROVIDER: str = "local"
    KG_API_BASE: str = "http://127.0.0.1:8002"
    KG_ENDPOINT: str = "/graph_query"
    KG_API_KEY: str = ""
    KG_TIMEOUT: int = 10
    STUDENT_DATA_PROVIDER: str = "local_csv_jsonl"
    SQLITE_DB_PATH: str = "outputs/local_student_data.db"
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = "teaching_agent"

    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def data_root_path(self) -> Path:
        return (self.PROJECT_ROOT / self.DATA_ROOT).resolve()

    @property
    def RAW_DIR(self) -> Path:
        return self.data_root_path / "raw"

    @property
    def MYSQL_DIR(self) -> Path:
        return self.data_root_path / "mysql"

    @property
    def SOURCES_DIR(self) -> Path:
        return self.data_root_path / "sources"

    @property
    def LABELS_DIR(self) -> Path:
        return self.data_root_path / "labels"

    @property
    def EVAL_DIR(self) -> Path:
        return self.data_root_path / "data"

    @property
    def output_dir_path(self) -> Path:
        return (self.PROJECT_ROOT / self.OUTPUT_DIR).resolve()


settings = Settings()
