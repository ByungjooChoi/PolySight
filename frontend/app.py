"""
PolySight - Agent Battle UI
Visual Agent (Jina V4 + MaxSim) vs Text Agent (Docling + BM25)
"""
import sys
import os
import asyncio
import uuid
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
from PIL import Image

# Set up paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Set HuggingFace cache
hf_cache_dir = os.path.join(project_root, "hf_cache")
os.environ["HF_HOME"] = hf_cache_dir
os.makedirs(hf_cache_dir, exist_ok=True)

load_dotenv()


def check_environment() -> dict:
    """
    Check all required environment settings and return status.
    Checks both .env file AND config.json (config.json takes priority).

    Returns:
        dict with status of each required setting
    """
    # Try to get config (reload to get latest values)
    try:
        from backend.utils.config_manager import get_config
        config = get_config()
        config.reload()  # Force reload to get latest config.json
        config_elastic_url = config.elastic_url
        config_elastic_api_key = config.elastic_api_key
        config_hf_token = config.hf_token
    except Exception:
        config_elastic_url = None
        config_elastic_api_key = None
        config_hf_token = None

    # Check both sources (config.json priority > .env)
    elastic_url = config_elastic_url or os.getenv("ELASTIC_CLOUD_SERVERLESS_URL")
    elastic_api_key = config_elastic_api_key or os.getenv("ELASTIC_API_KEY")
    hf_token = config_hf_token or os.getenv("HF_TOKEN")

    status = {
        "elastic_url": {
            "name": "ELASTIC_CLOUD_SERVERLESS_URL",
            "value": elastic_url,
            "required": True,
            "ok": False,
            "help": "Elastic Cloud Serverless 엔드포인트 URL"
        },
        "elastic_api_key": {
            "name": "ELASTIC_API_KEY",
            "value": elastic_api_key,
            "required": True,
            "ok": False,
            "help": "Elastic Cloud API 키"
        },
        "hf_token": {
            "name": "HF_TOKEN",
            "value": hf_token,
            "required": False,
            "ok": False,
            "help": "HuggingFace 토큰 (Jina V4 모델 다운로드용, 선택사항)"
        }
    }

    # Check each setting
    for key, info in status.items():
        value = info["value"]
        if value and value.strip() and not value.startswith("your-"):
            info["ok"] = True

    return status


def get_setup_status_html() -> str:
    """Generate HTML showing current setup status"""
    status = check_environment()

    all_required_ok = all(
        info["ok"] for info in status.values() if info["required"]
    )

    if all_required_ok:
        return ""  # Don't show anything if all is configured

    html = """
    <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <h3 style="color: #856404; margin-top: 0;">⚠️ 환경 설정이 필요합니다</h3>
        <p style="color: #856404;">PolySight를 사용하려면 <code>.env</code> 파일에 다음 설정이 필요합니다:</p>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #ffeeba;">
                <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ffc107;">설정</th>
                <th style="padding: 8px; text-align: center; border-bottom: 1px solid #ffc107;">상태</th>
                <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ffc107;">설명</th>
            </tr>
    """

    for key, info in status.items():
        icon = "✅" if info["ok"] else ("❌" if info["required"] else "⚪")
        color = "#28a745" if info["ok"] else ("#dc3545" if info["required"] else "#6c757d")
        required_badge = '<span style="background:#dc3545;color:white;padding:2px 6px;border-radius:3px;font-size:11px;margin-left:5px;">필수</span>' if info["required"] else '<span style="background:#6c757d;color:white;padding:2px 6px;border-radius:3px;font-size:11px;margin-left:5px;">선택</span>'

        html += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ffeeba;">
                    <code>{info["name"]}</code>{required_badge}
                </td>
                <td style="padding: 8px; text-align: center; border-bottom: 1px solid #ffeeba; color: {color};">{icon}</td>
                <td style="padding: 8px; border-bottom: 1px solid #ffeeba;">{info["help"]}</td>
            </tr>
        """

    html += """
        </table>
        <div style="margin-top: 12px; padding: 10px; background: #fff; border-radius: 4px;">
            <strong>설정 방법:</strong>
            <ol style="margin: 8px 0 0 0; padding-left: 20px;">
                <li>프로젝트 루트의 <code>.env.example</code>을 <code>.env</code>로 복사</li>
                <li><code>.env</code> 파일에 Elastic Cloud 정보 입력</li>
                <li>앱 재시작 (<code>python frontend/app.py</code>)</li>
            </ol>
        </div>
        <div style="margin-top: 10px; padding: 10px; background: #e7f3ff; border-radius: 4px;">
            <strong>💡 Elastic Cloud Serverless 계정이 없으신가요?</strong><br>
            <a href="https://cloud.elastic.co/registration" target="_blank">Elastic Cloud 무료 체험</a>에서 시작할 수 있습니다.
        </div>
    </div>
    """

    return html


def validate_environment_for_action(action_name: str) -> tuple[bool, str]:
    """
    Validate environment before performing an action.

    Returns:
        (is_valid, error_message)
    """
    status = check_environment()

    missing = []
    for key, info in status.items():
        if info["required"] and not info["ok"]:
            missing.append(f"- **{info['name']}**: {info['help']}")

    if missing:
        missing_text = "\n".join(missing)
        return False, f"""❌ **환경 설정 오류**

{action_name}을(를) 실행하려면 다음 설정이 필요합니다:

{missing_text}

**해결 방법:**
Settings 탭에서 Elasticsearch URL과 API Key를 입력하고 "설정 저장" 버튼을 클릭하세요."""

    return True, ""

# Import backend modules
from backend.pipelines.ingestion import IngestionManager, SearchManager
from backend.pipelines.visual_engine import process_uploaded_file
from backend.utils.elastic_client import ElasticClient
from backend.data.vidore_loader import ViDoReLoader
from backend.utils.config_manager import get_config, ConfigManager

# Temp directory for uploads
TEMP_DIR = os.path.join(project_root, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Global instances (lazy loaded)
_ingestion_manager: Optional[IngestionManager] = None
_search_manager: Optional[SearchManager] = None


def get_ingestion_manager() -> IngestionManager:
    global _ingestion_manager
    if _ingestion_manager is None:
        _ingestion_manager = IngestionManager()
    return _ingestion_manager


def get_search_manager() -> SearchManager:
    global _search_manager
    if _search_manager is None:
        _search_manager = SearchManager()
    return _search_manager


def format_result_card(result: dict, rank: int, agent_type: str) -> str:
    """Format a search result as HTML card"""
    score = result.get("score", 0)
    file_name = result.get("file_name", "Unknown")
    page_num = result.get("page_number", 0) + 1  # 1-indexed for display

    if agent_type == "visual":
        badge_color = "#4CAF50"
        badge_text = "Visual"
    else:
        badge_color = "#2196F3"
        badge_text = "Text"

    return f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 8px 0; background: #fafafa;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: bold; font-size: 16px;">#{rank}</span>
            <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{badge_text}</span>
        </div>
        <div style="margin-top: 8px;">
            <div><strong>File:</strong> {file_name}</div>
            <div><strong>Page:</strong> {page_num}</div>
            <div><strong>Score:</strong> {score:.4f}</div>
        </div>
    </div>
    """


def format_results_html(results: List[dict], agent_type: str, latency_ms: float) -> str:
    """Format all results as HTML"""
    if not results:
        return f"""
        <div style="text-align: center; padding: 20px; color: #666;">
            No results found
        </div>
        """

    header_color = "#4CAF50" if agent_type == "visual" else "#2196F3"
    agent_name = "Visual Agent (MaxSim)" if agent_type == "visual" else "Text Agent (BM25)"

    html = f"""
    <div style="border: 2px solid {header_color}; border-radius: 12px; overflow: hidden;">
        <div style="background: {header_color}; color: white; padding: 12px; text-align: center;">
            <h3 style="margin: 0;">{agent_name}</h3>
            <div style="font-size: 12px; opacity: 0.9;">Latency: {latency_ms:.1f}ms | Results: {len(results)}</div>
        </div>
        <div style="padding: 12px; max-height: 500px; overflow-y: auto;">
    """

    for i, result in enumerate(results, 1):
        html += format_result_card(result, i, agent_type)

    html += "</div></div>"
    return html


def search_agents(query: str, num_results: int = 5) -> Tuple[str, str]:
    """
    Search using both agents and return formatted results.

    Returns:
        Tuple of (visual_html, text_html)
    """
    if not query.strip():
        empty_msg = "<div style='text-align: center; padding: 40px; color: #999;'>Enter a query to search</div>"
        return empty_msg, empty_msg

    # Check environment first
    is_valid, error_msg = validate_environment_for_action("검색")
    if not is_valid:
        error_html = f"<div style='padding: 20px; color: #721c24; background: #f8d7da; border-radius: 8px;'>{error_msg.replace(chr(10), '<br>')}</div>"
        return error_html, error_html

    try:
        manager = get_search_manager()
        results = manager.search_both(query, size=num_results)

        visual_html = format_results_html(
            results["visual_agent"]["results"],
            "visual",
            results["visual_agent"]["latency_ms"]
        )

        text_html = format_results_html(
            results["text_agent"]["results"],
            "text",
            results["text_agent"]["latency_ms"]
        )

        return visual_html, text_html

    except Exception as e:
        error_html = f"""
        <div style="text-align: center; padding: 20px; color: #d32f2f;">
            <strong>Error:</strong> {str(e)}
        </div>
        """
        return error_html, error_html


def ingest_files(files: List[str], progress=gr.Progress()) -> str:
    """
    Ingest uploaded files through both pipelines.

    Args:
        files: List of file paths from gr.File

    Returns:
        Status message
    """
    if not files:
        return "No files uploaded"

    # Check environment first
    is_valid, error_msg = validate_environment_for_action("파일 인제스트")
    if not is_valid:
        return error_msg

    manager = get_ingestion_manager()
    results = []
    total = len(files)

    for i, file_path in enumerate(files):
        progress((i + 1) / total, desc=f"Processing {Path(file_path).name}...")

        try:
            # Process file
            stats = asyncio.run(manager.process_file(file_path))

            results.append(
                f"✅ **{stats['file_name']}**: "
                f"Visual={stats['visual_count']}, Text={stats['text_count']}, "
                f"Pages={stats['page_count']}"
            )

        except Exception as e:
            results.append(f"❌ **{Path(file_path).name}**: {str(e)}")

    # Summary
    success_count = sum(1 for r in results if r.startswith("✅"))
    summary = f"\n\n**Summary:** {success_count}/{total} files processed successfully"

    return "\n\n".join(results) + summary


def get_index_stats() -> str:
    """Get current index statistics"""
    try:
        client = ElasticClient()
        visual_count = client.get_index_count(ElasticClient.VISUAL_INDEX)
        text_count = client.get_index_count(ElasticClient.TEXT_INDEX)

        return f"""
### Index Statistics

| Index | Documents |
|-------|-----------|
| Visual (rank_vectors) | {visual_count} |
| Text (BM25) | {text_count} |
        """
    except Exception as e:
        return f"Error getting stats: {e}"


def clear_indices() -> str:
    """Clear all indices"""
    try:
        client = ElasticClient()
        client.clear_all_indices()
        return "✅ All indices cleared successfully"
    except Exception as e:
        return f"❌ Error clearing indices: {e}"


# ========== Settings Functions ==========

def load_current_settings() -> Tuple[str, str, str, str]:
    """Load current settings for UI display"""
    config = get_config()
    return (
        config.elastic_url or "",
        config.elastic_api_key or "",
        config.jina_api_key or "",
        config.hf_token or ""
    )


def save_settings(
    elastic_url: str,
    elastic_api_key: str,
    jina_api_key: str,
    hf_token: str
) -> str:
    """Save settings to config.json and reinitialize clients"""
    global _ingestion_manager, _search_manager

    config = get_config()

    config.set("elastic_url", elastic_url.strip())
    config.set("elastic_api_key", elastic_api_key.strip())
    config.set("jina_api_key", jina_api_key.strip())
    config.set("hf_token", hf_token.strip())

    if config.save():
        # Reset global managers to force reinitialization with new settings
        _ingestion_manager = None
        _search_manager = None

        # Also reset ElasticClient singleton
        try:
            from backend.utils.elastic_client import ElasticClient
            ElasticClient._instance = None
        except Exception:
            pass

        # Determine Jina mode
        jina_mode = "API 모드 ☁️" if jina_api_key.strip() else "로컬 모드 🖥️"
        return f"""✅ **설정이 저장되고 적용되었습니다!**

**현재 설정:**
- Elasticsearch: {'✅ 설정됨' if elastic_url and elastic_api_key else '❌ 미설정'}
- Jina V4: {jina_mode}
- HuggingFace: {'✅ 설정됨' if hf_token else '⚪ 미설정 (선택사항)'}

✅ **재시작 없이 바로 사용 가능합니다!**"""
    else:
        return "❌ 설정 저장 실패. 로그를 확인하세요."


def test_elastic_connection(url: str, api_key: str) -> str:
    """Test Elasticsearch connection"""
    if not url or not api_key:
        return "❌ URL과 API Key를 모두 입력하세요."

    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(
            url.strip(),
            api_key=api_key.strip(),
            request_timeout=10
        )
        info = es.info()
        cluster_name = info.get("cluster_name", "Unknown")
        version = info.get("version", {}).get("number", "Unknown")

        return f"""✅ **연결 성공!**

- Cluster: {cluster_name}
- Version: {version}"""
    except Exception as e:
        return f"❌ **연결 실패:** {str(e)}"


def test_jina_api(api_key: str) -> str:
    """Test Jina API connection"""
    if not api_key:
        return "ℹ️ API Key가 비어있으면 로컬 모드로 동작합니다."

    try:
        import requests

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "jina-embeddings-v4",
            "task": "retrieval.query",
            "input": ["test"],
            "embedding_type": "float"
        }

        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=15
        )

        if response.status_code == 200:
            return "✅ **Jina API 연결 성공!** API 모드로 사용 가능합니다."
        else:
            error = response.json().get("detail", response.text)
            return f"❌ **API 오류:** {error}"

    except Exception as e:
        return f"❌ **연결 실패:** {str(e)}"


def get_current_config_status() -> str:
    """Get formatted current configuration status"""
    config = get_config()
    status = config.get_status()

    elastic_status = "✅ 연결됨" if status["elastic"]["configured"] else "❌ 미설정"
    jina_mode = "☁️ API 모드" if status["jina"]["api_configured"] else "🖥️ 로컬 모드"
    hf_status = "✅ 설정됨" if status["hf"]["configured"] else "⚪ 미설정"

    return f"""### 현재 상태

| 항목 | 상태 |
|------|------|
| Elasticsearch | {elastic_status} |
| Jina V4 | {jina_mode} |
| HuggingFace Token | {hf_status} |
"""


def check_vidore_loaded() -> Tuple[bool, int]:
    """Check if ViDoRe samples are already loaded"""
    try:
        client = ElasticClient()
        # Check for documents with vidore prefix in doc_id
        result = client.es.count(
            index=ElasticClient.VISUAL_INDEX,
            query={"prefix": {"doc_id": "vidore_"}}
        )
        count = result.get("count", 0)
        return count > 0, count
    except Exception:
        return False, 0


def load_vidore_samples(num_samples: int = 20, progress=gr.Progress()) -> str:
    """
    Load ViDoRe benchmark samples into the index.
    Checks for duplicates before loading.
    """
    # Check environment first
    is_valid, error_msg = validate_environment_for_action("ViDoRe 샘플 로드")
    if not is_valid:
        return error_msg

    # Check if already loaded
    already_loaded, existing_count = check_vidore_loaded()
    if already_loaded:
        return f"⚠️ ViDoRe 샘플이 이미 로드되어 있습니다! (현재 {existing_count}개 문서)\n\n다시 로드하려면 먼저 'Settings' 탭에서 인덱스를 초기화하세요."

    try:
        progress(0.1, desc="ViDoRe 데이터셋 로딩 중...")
        loader = ViDoReLoader()
        samples = loader.get_samples("test", num_samples)

        if not samples:
            return "❌ ViDoRe 샘플을 가져올 수 없습니다."

        manager = get_ingestion_manager()
        success_count = 0
        total = len(samples)

        for i, sample in enumerate(samples):
            progress((i + 1) / total * 0.9 + 0.1, desc=f"인덱싱 중... {i+1}/{total}")

            image = sample.get("image")
            if image is None:
                continue

            try:
                # Create unique doc_id with vidore prefix
                doc_id = f"vidore_{sample.get('doc_id', f'doc_{i}')}_{sample.get('page_id', 0)}"

                # Process through both pipelines
                asyncio.run(manager.process_image(
                    image=image,
                    doc_id=doc_id,
                    page_number=sample.get("page_id", 0),
                    file_name=f"vidore_sample_{i}.png"
                ))
                success_count += 1

            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")

        # Get sample queries for user
        sample_queries = loader.get_queries("test")[:5]
        queries_text = "\n".join([f"  • {q}" for q in sample_queries])

        return f"""✅ ViDoRe 샘플 로드 완료!

**결과:** {success_count}/{total} 샘플 인덱싱 성공

**검색 예시 쿼리:**
{queries_text}

Search Battle 탭에서 위 쿼리로 검색해보세요!"""

    except Exception as e:
        return f"❌ 에러 발생: {str(e)}"


# ========== Gradio UI ==========

with gr.Blocks(
    title="PolySight - Agent Battle",
    theme=gr.themes.Soft(),
    css="""
    .result-container { min-height: 400px; }
    .header-text { text-align: center; margin-bottom: 20px; }
    """
) as app:

    # Header
    gr.Markdown(
        """
        # 🔍 PolySight: Agent Battle

        **Visual Agent** (Jina V4 Multi-vector + MaxSim) **vs** **Text Agent** (Docling OCR + BM25)

        Compare Late Interaction visual search against traditional OCR-based text search.
        """,
        elem_classes=["header-text"]
    )

    # Environment Status Banner (shows only if config is missing)
    setup_status = get_setup_status_html()
    if setup_status:
        gr.HTML(setup_status)

    with gr.Tabs():
        # Tab 1: Search (Agent Battle)
        with gr.TabItem("🎯 Search Battle", id="search"):
            with gr.Row():
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query...",
                    scale=4
                )
                num_results = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Results",
                    scale=1
                )
                search_btn = gr.Button("🔍 Search", variant="primary", scale=1)

            with gr.Row(equal_height=True):
                visual_results = gr.HTML(
                    label="Visual Agent Results",
                    elem_classes=["result-container"]
                )
                text_results = gr.HTML(
                    label="Text Agent Results",
                    elem_classes=["result-container"]
                )

            # Search event handlers
            search_btn.click(
                fn=search_agents,
                inputs=[query_input, num_results],
                outputs=[visual_results, text_results]
            )
            query_input.submit(
                fn=search_agents,
                inputs=[query_input, num_results],
                outputs=[visual_results, text_results]
            )

        # Tab 2: Ingest Documents
        with gr.TabItem("📤 Ingest Documents", id="ingest"):
            gr.Markdown(
                """
                ### Upload Documents

                Upload PDF files or images to index them through both pipelines:
                - **Visual Pipeline**: Image → Jina V4 Multi-vector → Token Pooling → Elastic (rank_vectors)
                - **Text Pipeline**: Image → Docling OCR → Text → Elastic (BM25)
                """
            )

            # ViDoRe Sample Loader Section
            gr.Markdown("---")
            gr.Markdown("### 🎯 Quick Start: Load Demo Data")

            with gr.Row():
                vidore_samples_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Number of ViDoRe Samples",
                    scale=2
                )
                vidore_btn = gr.Button("📥 Load ViDoRe Samples", variant="secondary", scale=1)

            vidore_output = gr.Markdown(label="ViDoRe Load Results")

            vidore_btn.click(
                fn=load_vidore_samples,
                inputs=[vidore_samples_slider],
                outputs=[vidore_output]
            )

            gr.Markdown("---")
            gr.Markdown("### 📁 Upload Custom Files")

            with gr.Row():
                file_upload = gr.File(
                    label="Upload Files",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff"],
                    file_count="multiple",
                    scale=2
                )

            with gr.Row():
                ingest_btn = gr.Button("🚀 Start Ingestion", variant="primary")

            ingest_output = gr.Markdown(label="Ingestion Results")

            ingest_btn.click(
                fn=ingest_files,
                inputs=[file_upload],
                outputs=[ingest_output]
            )

        # Tab 3: Settings & Stats
        with gr.TabItem("⚙️ Settings", id="settings"):

            # Current Status
            config_status_display = gr.Markdown(value=get_current_config_status())

            gr.Markdown("---")

            # Elasticsearch Settings
            gr.Markdown("### 🔌 Elasticsearch 설정")
            with gr.Row():
                elastic_url_input = gr.Textbox(
                    label="Elastic Cloud Serverless URL",
                    placeholder="https://your-deployment.es.region.aws.elastic.cloud",
                    type="text",
                    scale=3
                )
                elastic_test_btn = gr.Button("🔗 연결 테스트", scale=1)

            elastic_api_key_input = gr.Textbox(
                label="Elastic API Key",
                placeholder="API Key를 입력하세요",
                type="password"
            )
            elastic_test_output = gr.Markdown()

            elastic_test_btn.click(
                fn=test_elastic_connection,
                inputs=[elastic_url_input, elastic_api_key_input],
                outputs=[elastic_test_output]
            )

            gr.Markdown("---")

            # Jina V4 Settings
            gr.Markdown("### 🤖 Jina V4 설정")
            gr.Markdown("""
            **모드 선택:**
            - **로컬 모드 (기본)**: GPU 권장, 무료
            - **API 모드**: API Key 입력 시 자동 전환, GPU 불필요
            """)

            with gr.Row():
                jina_api_key_input = gr.Textbox(
                    label="Jina API Key (선택 - 입력 시 API 모드로 전환)",
                    placeholder="jina_xxxxxxxxxxxxxxxx (비워두면 로컬 모드)",
                    type="password",
                    scale=3
                )
                jina_test_btn = gr.Button("🔗 API 테스트", scale=1)

            jina_test_output = gr.Markdown()

            jina_test_btn.click(
                fn=test_jina_api,
                inputs=[jina_api_key_input],
                outputs=[jina_test_output]
            )

            gr.Markdown("---")

            # HuggingFace Settings
            gr.Markdown("### 🤗 HuggingFace 설정 (선택)")
            hf_token_input = gr.Textbox(
                label="HuggingFace Token (로컬 모델 다운로드 시 필요할 수 있음)",
                placeholder="hf_xxxxxxxxxxxxxxxx",
                type="password"
            )

            gr.Markdown("---")

            # Save Button
            with gr.Row():
                save_btn = gr.Button("💾 설정 저장", variant="primary", scale=2)
                reload_btn = gr.Button("🔄 현재 설정 불러오기", scale=1)

            save_output = gr.Markdown()

            save_btn.click(
                fn=save_settings,
                inputs=[elastic_url_input, elastic_api_key_input, jina_api_key_input, hf_token_input],
                outputs=[save_output]
            )

            reload_btn.click(
                fn=load_current_settings,
                outputs=[elastic_url_input, elastic_api_key_input, jina_api_key_input, hf_token_input]
            )

            gr.Markdown("---")

            # Index Statistics & Danger Zone
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Index Statistics")
                    stats_output = gr.Markdown()
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats")

                    refresh_stats_btn.click(
                        fn=get_index_stats,
                        outputs=[stats_output]
                    )

                with gr.Column():
                    gr.Markdown("### ⚠️ Danger Zone")
                    gr.Markdown("모든 인덱싱된 문서가 삭제됩니다!")
                    clear_btn = gr.Button("🗑️ Clear All Indices", variant="stop")
                    clear_output = gr.Markdown()

                    clear_btn.click(
                        fn=clear_indices,
                        outputs=[clear_output]
                    )

            # Load current settings on page load
            app.load(
                fn=load_current_settings,
                outputs=[elastic_url_input, elastic_api_key_input, jina_api_key_input, hf_token_input]
            )

    # Footer
    gr.Markdown(
        """
        ---
        **PolySight** | Powered by Elastic Cloud Serverless & Jina V4 | Late Interaction (MaxSim) Demo
        """,
        elem_classes=["header-text"]
    )


# Launch
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
