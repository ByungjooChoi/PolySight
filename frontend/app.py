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
from datetime import datetime

import gradio as gr

# Set up paths first
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Setup logging to file
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"polysight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8')
        # Console output removed - logs go to file only
    ]
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from PIL import Image

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


def get_ingestion_manager(pool_factor: int = 3) -> IngestionManager:
    global _ingestion_manager
    if _ingestion_manager is None:
        _ingestion_manager = IngestionManager(pool_factor=pool_factor)
    else:
        # Update pool_factor if changed
        if _ingestion_manager.pool_factor != pool_factor:
            _ingestion_manager.pool_factor = pool_factor
    return _ingestion_manager


def get_search_manager() -> SearchManager:
    global _search_manager
    if _search_manager is None:
        _search_manager = SearchManager()
    return _search_manager


def get_image_base64(image_path: str, max_size: tuple = (300, 400)) -> str:
    """Convert image to base64 thumbnail for HTML embedding."""
    import base64
    from io import BytesIO

    if not image_path or not os.path.exists(image_path):
        return ""

    try:
        with Image.open(image_path) as img:
            # Create thumbnail
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG", optimize=True)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return ""


def get_full_image_base64(image_path: str) -> str:
    """Convert full-size image to base64 for modal display."""
    import base64
    from io import BytesIO

    if not image_path or not os.path.exists(image_path):
        return ""

    try:
        with Image.open(image_path) as img:
            # Keep original size but limit to reasonable max for web display
            max_dimension = 1600
            if img.width > max_dimension or img.height > max_dimension:
                ratio = min(max_dimension / img.width, max_dimension / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG", quality=95)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logger.warning(f"Failed to load full image {image_path}: {e}")
        return ""


def format_explanation_summary(explanation: dict, agent_type: str) -> str:
    """Format Elasticsearch explanation into readable summary."""
    if not explanation:
        return ""

    try:
        if agent_type == "visual":
            # MaxSim explanation - extract key info
            desc = explanation.get("description", "")
            if "maxSimDotProduct" in desc:
                return "Late Interaction: max(query_token · doc_token) summed across all query tokens"
            return "MaxSim multi-vector similarity"
        else:
            # BM25 explanation - extract term frequencies
            desc = explanation.get("description", "")
            details = explanation.get("details", [])

            # Try to extract useful BM25 info
            terms_info = []
            for detail in details:
                detail_desc = detail.get("description", "")
                if "weight(" in detail_desc:
                    # Extract term and field
                    import re
                    match = re.search(r'weight\(([^:]+):([^)]+)', detail_desc)
                    if match:
                        field, term = match.groups()
                        term_score = detail.get("value", 0)
                        terms_info.append(f'"{term.strip()}"={term_score:.2f}')

            if terms_info:
                return f"BM25 term scores: {', '.join(terms_info[:3])}"
            return "BM25 (term frequency × inverse document frequency)"
    except Exception:
        pass

    return ""


def format_result_card(result: dict, rank: int, agent_type: str) -> str:
    """Format a search result as HTML card with image preview (colpali style)"""
    score = float(result.get("score", 0) or 0)
    raw_score = result.get("raw_score")  # For visual search, this is the unnormalized score
    file_name = str(result.get("file_name", "Unknown"))
    page_num_raw = result.get("page_number", 0)
    page_num = (int(page_num_raw) if page_num_raw is not None else 0) + 1  # 1-indexed for display
    image_path = result.get("image_path")
    highlight = result.get("highlight", "")
    explanation = result.get("explanation", {})

    if agent_type == "visual":
        badge_color = "#4CAF50"
        badge_text = "Visual"
        # Show normalized score with raw score in tooltip
        if raw_score is not None:
            score_label = f"Score (raw: {raw_score:.2f})"
        else:
            score_label = "MaxSim"
    else:
        badge_color = "#2196F3"
        badge_text = "Text"
        score_label = "BM25"

    # Get image thumbnail (larger for better visibility like colpali)
    img_base64 = get_image_base64(image_path) if image_path else ""
    # Get full-size image for modal
    full_img_base64 = get_full_image_base64(image_path) if image_path else ""

    # Unique ID for this card's modal
    modal_id = f"modal_{agent_type}_{rank}_{hash(str(image_path)) % 10000}"

    # Image HTML - clickable thumbnail that opens modal with full-size image
    if img_base64:
        image_html = f'''
        <div style="text-align: center; margin-bottom: 10px;">
            <img src="{img_base64}"
                 onclick="document.getElementById('{modal_id}').style.display='flex'"
                 style="max-width: 100%; max-height: 200px; border: 1px solid #ddd; border-radius: 4px; object-fit: contain; box-shadow: 0 2px 4px rgba(0,0,0,0.1); cursor: pointer; transition: transform 0.2s, box-shadow 0.2s;"
                 onmouseover="this.style.transform='scale(1.02)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)'"
                 onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'"
                 title="클릭하여 원본 보기" />
            <div style="font-size: 10px; color: #888; margin-top: 4px;">🔍 클릭하여 확대</div>
        </div>
        <!-- Modal for full-size image -->
        <div id="{modal_id}" onclick="if(event.target===this)this.style.display='none'"
             style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 9999; justify-content: center; align-items: center; cursor: pointer;">
            <div style="position: relative; max-width: 90%; max-height: 90%; background: white; border-radius: 8px; padding: 10px; box-shadow: 0 20px 60px rgba(0,0,0,0.5);">
                <button onclick="document.getElementById('{modal_id}').style.display='none'"
                        style="position: absolute; top: -12px; right: -12px; width: 32px; height: 32px; border-radius: 50%; border: none; background: #ff4444; color: white; font-size: 18px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.3); z-index: 10000;">✕</button>
                <img src="{full_img_base64}" style="max-width: 85vw; max-height: 80vh; object-fit: contain; border-radius: 4px;" />
                <div style="text-align: center; padding: 10px; color: #333; font-size: 14px;">
                    <strong>{file_name}</strong> | Page {page_num} | {score_label}: {score:.4f}
                </div>
            </div>
        </div>
        '''
    else:
        image_html = f'''
        <div style="text-align: center; margin-bottom: 10px;">
            <div style="width: 100%; height: 100px; background: #f0f0f0; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px;">
                📄 No Preview
            </div>
        </div>
        '''

    # Highlight text for BM25 results
    highlight_html = ""
    if highlight and agent_type == "text":
        highlight_html = f'''
        <div style="margin-top: 8px; padding: 6px 8px; background: #fffbeb; border-radius: 4px; font-size: 11px; border-left: 3px solid #f59e0b;">
            {highlight}
        </div>
        '''

    # Explanation summary
    explain_summary = format_explanation_summary(explanation, agent_type)
    explain_html = ""
    if explain_summary:
        explain_html = f'''
        <div style="font-size: 10px; color: #666; margin-top: 4px; padding: 4px 6px; background: #f5f5f5; border-radius: 3px;">
            💡 {explain_summary}
        </div>
        '''

    return f"""
    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: bold; font-size: 18px; color: #333;">#{rank}</span>
            <span style="background: {badge_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;">{badge_text}</span>
        </div>
        {image_html}
        <div style="text-align: center;">
            <div style="font-size: 12px; color: #555; margin-bottom: 4px;">{file_name}</div>
            <div style="font-size: 11px; color: #888;">Page {page_num}</div>
            <div style="margin-top: 8px; padding: 6px 12px; background: linear-gradient(135deg, {badge_color}22, {badge_color}11); border-radius: 6px; display: inline-block;">
                <span style="font-size: 11px; color: #666;">{score_label}</span>
                <span style="font-size: 16px; font-weight: bold; color: {badge_color}; margin-left: 4px;">{score:.4f}</span>
            </div>
        </div>
        {highlight_html}
        {explain_html}
    </div>
    """


def format_results_html(results: List[dict], agent_type: str, latency_ms: float) -> str:
    """Format all results as HTML grid (colpali style)"""
    if not results:
        return f"""
        <div style="text-align: center; padding: 40px; color: #666;">
            No results found
        </div>
        """

    header_color = "#4CAF50" if agent_type == "visual" else "#2196F3"
    agent_name = "🔍 Visual Agent (MaxSim)" if agent_type == "visual" else "📝 Text Agent (BM25)"

    # Grid layout like colpali demo
    html = f"""
    <div style="border: 2px solid {header_color}; border-radius: 12px; overflow: hidden;">
        <div style="background: linear-gradient(135deg, {header_color}, {header_color}dd); color: white; padding: 14px; text-align: center;">
            <h3 style="margin: 0; font-size: 16px;">{agent_name}</h3>
            <div style="font-size: 12px; opacity: 0.9; margin-top: 4px;">
                ⚡ {latency_ms:.1f}ms | 📊 {len(results)} results
            </div>
        </div>
        <div style="padding: 12px; background: #fafafa;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
    """

    for i, result in enumerate(results, 1):
        html += format_result_card(result, i, agent_type)

    html += "</div></div></div>"
    return html


def search_agents(
    query: str,
    num_results: int = 5,
    visual_threshold: float = 0.0,
    normalize_scores: bool = True
) -> Tuple[str, str]:
    """
    Search using both agents and return formatted results.

    Args:
        query: Search query text
        num_results: Number of results to return
        visual_threshold: Minimum normalized score for visual results (0=no filter)
        normalize_scores: Whether to normalize visual scores

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

        # Convert threshold: 0.0 means no filtering
        threshold = visual_threshold if visual_threshold > 0 else None

        results = manager.search_both(
            query,
            size=num_results,
            normalize_visual=normalize_scores,
            visual_threshold=threshold
        )

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


def ingest_files(files: List[str], pool_factor: int = 3, progress=gr.Progress()) -> str:
    """
    Ingest uploaded files through both pipelines.

    Args:
        files: List of file paths from gr.File
        pool_factor: Token pooling factor

    Returns:
        Status message
    """
    if not files:
        return "No files uploaded"

    # Check environment first
    is_valid, error_msg = validate_environment_for_action("파일 인제스트")
    if not is_valid:
        return error_msg

    manager = get_ingestion_manager(pool_factor=pool_factor)
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


# Default sample queries - ViDoRe V3 공식 쿼리 및 샘플 문서 매칭 쿼리
# 우리가 인덱싱한 샘플 문서들을 검색할 수 있는 쿼리들
DEFAULT_SAMPLE_QUERIES = [
    # ViDoRe V3 HR 공식 쿼리 (query_id=4) - ground truth: corpus_id 12, 25
    "estimated skilled labor needs for EU green transition by 2030",
    # HR 도메인 추가 쿼리
    "European Green Deal employment impact",
    "skills investment needs for green transition",
    # Finance 도메인
    "JPMorgan Chase financial performance 2024",
    # CS 도메인
    "Python programming basics",
]


def generate_sample_queries_html(queries: list) -> str:
    """Generate HTML for clickable sample query buttons."""
    if not queries:
        return ""

    buttons = []
    for q in queries:
        # Escape special characters for safe HTML/JS embedding
        safe_query = q.replace("'", "\\'").replace('"', "&quot;")

        # Use single quotes for onclick attribute and escape properly
        onclick = (
            f"(function(btn){{"
            f"var q=btn.getAttribute(&apos;data-query&apos;);"
            f"var inp=document.querySelector(&apos;#search-query-input textarea&apos;)||document.querySelector(&apos;#search-query-input input&apos;);"
            f"if(inp){{"
            f"inp.value=q;"
            f"inp.dispatchEvent(new Event(&apos;input&apos;,{{bubbles:true}}));"
            f"setTimeout(function(){{var bs=document.querySelectorAll(&apos;button&apos;);for(var i=0;i<bs.length;i++){{if(bs[i].textContent.indexOf(&apos;Search&apos;)>=0){{bs[i].click();break;}}}}}},100);"
            f"}}"
            f"}})(this)"
        )

        btn = (
            f'<button onclick="{onclick}" data-query="{safe_query}" '
            f'style="margin: 4px; padding: 6px 12px; border: 1px solid #ddd; '
            f'border-radius: 16px; background: #f5f5f5; cursor: pointer; '
            f'font-size: 13px; transition: all 0.2s;" '
            f'onmouseover="this.style.background=\'#e0e0e0\'" '
            f'onmouseout="this.style.background=\'#f5f5f5\'">{q}</button>'
        )
        buttons.append(btn)

    buttons_html = " ".join(buttons)

    return f"""
    <div style="margin: 10px 0; padding: 10px; background: #fafafa; border-radius: 8px;">
        <span style="font-weight: bold; margin-right: 10px;">💡 예시 쿼리:</span>
        {buttons_html}
    </div>
    """


SAMPLE_QUERIES_FILE = Path(__file__).parent.parent / "data" / "sample_queries.json"


def save_sample_queries(queries: list):
    """Save sample queries to file for persistence."""
    try:
        SAMPLE_QUERIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(SAMPLE_QUERIES_FILE, "w") as f:
            json.dump(queries, f)
    except Exception as e:
        logger.warning(f"Failed to save sample queries: {e}")


def load_saved_sample_queries() -> list:
    """Load sample queries from file."""
    try:
        import json
        if SAMPLE_QUERIES_FILE.exists():
            with open(SAMPLE_QUERIES_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def get_initial_sample_queries_html() -> str:
    """Get sample queries HTML if documents exist in index."""
    try:
        from backend.utils.elastic_client import ElasticClient
        client = ElasticClient()
        # Check if there are documents in visual index
        if client.client:
            count = client.client.count(index=client.VISUAL_INDEX).get("count", 0)
            if count > 0:
                # Try to load saved queries from ViDoRe, fall back to defaults
                queries = load_saved_sample_queries()
                if not queries:
                    queries = DEFAULT_SAMPLE_QUERIES
                return generate_sample_queries_html(queries)
    except Exception:
        pass
    return ""


def load_vidore_samples(pool_factor: int = 3, progress=gr.Progress(track_tqdm=True)) -> tuple[str, str]:
    """
    Load ViDoRe V3 benchmark samples into the index.
    Loads 100 samples from each of 8 datasets (800 total).

    Args:
        pool_factor: Token pooling factor (1=no pooling, 2-5 recommended)

    Returns:
        tuple: (status_message, sample_queries_html)
    """
    # V3 datasets configuration
    V3_DATASETS = [
        ("vidore/vidore_v3_hr", "hr", "EU HR 문서"),
        ("vidore/vidore_v3_finance_en", "finance_en", "금융 (영어)"),
        ("vidore/vidore_v3_industrial", "industrial", "항공기 기술문서"),
        ("vidore/vidore_v3_pharmaceuticals", "pharma", "제약 문서"),
        ("vidore/vidore_v3_computer_science", "cs", "CS 교과서"),
        ("vidore/vidore_v3_energy", "energy", "에너지 보고서"),
        ("vidore/vidore_v3_physics", "physics", "물리학 슬라이드"),
        ("vidore/vidore_v3_finance_fr", "finance_fr", "금융 (프랑스어)"),
    ]
    SAMPLES_PER_DATASET = 100
    TOTAL_SAMPLES = len(V3_DATASETS) * SAMPLES_PER_DATASET

    # Initialize progress bar immediately (fixes first-run issue)
    progress(0, desc="시작 중...")

    # Check environment first
    is_valid, error_msg = validate_environment_for_action("ViDoRe V3 샘플 로드")
    if not is_valid:
        return error_msg, ""

    progress(0.02, desc="환경 확인 완료...")

    # Check if already loaded
    already_loaded, existing_count = check_vidore_loaded()
    if already_loaded:
        return f"⚠️ ViDoRe 샘플이 이미 로드되어 있습니다! (현재 {existing_count}개 문서)\n\n다시 로드하려면 먼저 'Settings' 탭에서 인덱스를 초기화하세요.", ""

    try:
        from datasets import load_dataset
        import time

        manager = get_ingestion_manager(pool_factor=pool_factor)
        logger.info(f"Loading ViDoRe V3 with pool_factor={pool_factor}")
        success_count = 0
        dataset_stats = {}
        global_idx = 0

        for ds_idx, (dataset_name, domain, domain_desc) in enumerate(V3_DATASETS):
            progress(ds_idx / len(V3_DATASETS), desc=f"[{ds_idx+1}/{len(V3_DATASETS)}] {domain_desc} 로딩 중...")

            try:
                # V3 datasets require config name 'corpus' for documents
                corpus = load_dataset(
                    dataset_name,
                    "corpus",
                    split="test",  # V3 corpus uses 'test' split
                    streaming=True
                )

                # Collect batch data for this dataset
                batch_data = []
                progress(ds_idx / len(V3_DATASETS), desc=f"[{ds_idx+1}/{len(V3_DATASETS)}] {domain_desc}: 샘플 수집 중...")

                for i, sample in enumerate(corpus.take(SAMPLES_PER_DATASET)):
                    # V3 corpus fields: image, doc_id, page_number_in_doc
                    image = sample.get("image")
                    if image is None:
                        continue

                    # V3 corpus fields (from HuggingFace viewer)
                    raw_doc_id = sample.get('doc_id', f'doc_{i}')
                    page_num = sample.get('page_number_in_doc', 0)
                    doc_id = f"v3_{domain}_{raw_doc_id}_{page_num}"

                    batch_data.append({
                        "image": image,
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "file_name": f"v3_{domain}_{i}.png"
                    })
                    global_idx += 1

                # Process batch
                if batch_data:
                    progress(
                        (ds_idx + 0.5) / len(V3_DATASETS),
                        desc=f"[{ds_idx+1}/{len(V3_DATASETS)}] {domain_desc}: 배치 처리 중 ({len(batch_data)}개)..."
                    )

                    def batch_progress(current, total, desc):
                        p = (ds_idx + current / total) / len(V3_DATASETS)
                        progress(p, desc=f"[{ds_idx+1}/{len(V3_DATASETS)}] {domain_desc}: {current}/{total}")

                    try:
                        results = asyncio.run(manager.process_images_batch(
                            batch_data,
                            progress_callback=batch_progress
                        ))
                        ds_success = sum(1 for r in results if r.get("visual_indexed") or r.get("text_indexed"))
                        success_count += ds_success
                    except Exception as e:
                        logger.error(f"Batch processing failed for {domain}: {e}")
                        ds_success = 0

                dataset_stats[domain_desc] = ds_success
                logger.info(f"Loaded {ds_success} samples from {dataset_name}")

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                dataset_stats[domain_desc] = 0

        # 항상 우리가 정의한 샘플 문서 매칭 쿼리 사용
        sample_queries = DEFAULT_SAMPLE_QUERIES[:5]
        save_sample_queries(sample_queries)

        queries_text = "\n".join([f"  • {q[:60]}..." if len(q) > 60 else f"  • {q}" for q in sample_queries])

        # Format dataset stats
        stats_text = "\n".join([f"  • {domain}: {count}개" for domain, count in dataset_stats.items()])

        result_msg = f"""✅ ViDoRe V3 벤치마크 인덱싱 완료!

**결과:** {success_count}/{TOTAL_SAMPLES} 샘플 인덱싱 성공

**데이터셋별 현황:**
{stats_text}

**검색 예시 쿼리:**
{queries_text}

Search Battle 탭에서 위 쿼리로 검색해보세요!"""

        # Generate HTML for sample query buttons
        sample_queries_html = generate_sample_queries_html(sample_queries)

        return result_msg, sample_queries_html

    except ImportError:
        return "❌ datasets 라이브러리가 필요합니다. `pip install datasets` 실행하세요.", ""
    except Exception as e:
        return f"❌ 에러 발생: {str(e)}", ""


# ========== Gradio UI ==========

CUSTOM_CSS = """
.result-container { min-height: 400px; }
.header-text { text-align: center; margin-bottom: 20px; }
.sample-query-btn { margin: 2px !important; }
.sample-queries-row { margin-top: 10px !important; }
"""

with gr.Blocks(title="PolySight - Agent Battle") as app:

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

    # State for sample queries (populated when ViDoRe is loaded)
    sample_queries_state = gr.State([])

    with gr.Tabs():
        # Tab 1: Search (Agent Battle)
        with gr.TabItem("🎯 Search Battle", id="search"):
            with gr.Row():
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query...",
                    scale=4,
                    elem_id="search-query-input"
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

            # Advanced search options (collapsible)
            with gr.Accordion("⚙️ Advanced Options", open=False):
                with gr.Row():
                    visual_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="Visual Score Threshold",
                        info="0=필터없음, 0.5=중간 유사도 이상만, 0.7=높은 유사도만"
                    )
                    normalize_scores = gr.Checkbox(
                        label="Normalize Scores",
                        value=True,
                        info="쿼리 길이와 무관하게 0-1 범위로 정규화"
                    )

            # Sample query buttons container (shows default queries if documents exist)
            sample_queries_html = gr.HTML(
                value=get_initial_sample_queries_html(),
                visible=True,
                elem_id="sample-queries-container"
            )

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
                inputs=[query_input, num_results, visual_threshold, normalize_scores],
                outputs=[visual_results, text_results]
            )
            query_input.submit(
                fn=search_agents,
                inputs=[query_input, num_results, visual_threshold, normalize_scores],
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
            gr.Markdown("### ⚙️ Embedding Settings")

            with gr.Row():
                pool_factor_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Token Pooling Factor",
                    info="1=풀링없음, 3=기본값(~94% 정확도), 높을수록 벡터 수 감소",
                    scale=2
                )
            pool_factor_display = gr.Markdown(
                value="**현재 설정:** pool_factor=3 (벡터 수 ~1/3로 감소)"
            )

            def update_pool_factor_display(factor):
                if factor == 1:
                    return "**현재 설정:** pool_factor=1 (풀링 없음, 최대 정확도)"
                else:
                    reduction = f"~1/{factor}"
                    accuracy = {2: "~97%", 3: "~94%", 4: "~91%", 5: "~88%"}.get(factor, "~90%")
                    return f"**현재 설정:** pool_factor={factor} (벡터 수 {reduction}로 감소, 정확도 {accuracy})"

            pool_factor_slider.change(
                fn=update_pool_factor_display,
                inputs=[pool_factor_slider],
                outputs=[pool_factor_display]
            )

            gr.Markdown("---")
            gr.Markdown("### 🎯 Quick Start: Load Demo Data")

            gr.Markdown("""
**ViDoRe Benchmark V3** - 엔터프라이즈 문서 검색 벤치마크

8개 도메인에서 각 100개씩, 총 800개 샘플을 로드합니다:
- 🏢 HR (EU 행정문서) · 💰 Finance EN/FR (금융)
- ✈️ Industrial (항공기 기술) · 💊 Pharmaceuticals (제약)
- 💻 Computer Science (CS 교과서) · ⚡ Energy (에너지 보고서)
- 🔬 Physics (물리학 슬라이드)
""")

            with gr.Row():
                vidore_btn = gr.Button("📥 Load ViDoRe V3 Samples (800개)", variant="secondary", scale=1)

            vidore_output = gr.Markdown(label="ViDoRe Load Results")
            # Hidden state to pass sample queries HTML between callbacks
            vidore_queries_state = gr.State("")

            # Step 1: Run main ingestion with progress (only update Ingest tab components)
            # Step 2: Update Search tab's sample_queries_html via .then() chain
            vidore_btn.click(
                fn=load_vidore_samples,
                inputs=[pool_factor_slider],
                outputs=[vidore_output, vidore_queries_state],
                show_progress="full"
            ).then(
                fn=lambda x: x,  # Pass through the HTML
                inputs=[vidore_queries_state],
                outputs=[sample_queries_html]
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
                inputs=[file_upload, pool_factor_slider],
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
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )
