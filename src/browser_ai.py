"""Browser-side open-model assistant UI powered by WebLLM."""
from __future__ import annotations
import json
from pathlib import Path

THEMES = {
    "Light": {"bg":"#f7f9fc","panel":"#ffffff","text":"#172b4d","muted":"#64748b","accent":"#2563eb","line":"#e2e8f0"},
    "Dark": {"bg":"#101827","panel":"#182437","text":"#e5edf8","muted":"#a8b6c9","accent":"#60a5fa","line":"#34445a"},
    "Green": {"bg":"#eef6f0","panel":"#ffffff","text":"#173b2b","muted":"#557363","accent":"#16845b","line":"#d7e7dc"},
    "Ocean": {"bg":"#edf5fb","panel":"#ffffff","text":"#15324b","muted":"#58758d","accent":"#087eae","line":"#d6e6f0"},
    "Purple": {"bg":"#f4f0fb","panel":"#ffffff","text":"#30234d","muted":"#76678f","accent":"#7652c8","line":"#e5dcf2"},
    "Sand": {"bg":"#f8f4ec","panel":"#fffdf8","text":"#43372b","muted":"#847766","accent":"#a66a32","line":"#e9dfd0"},
}

def build_browser_ai_html(dataset_context: str = "", theme: str = "Light") -> str:
    """Build the embedded browser-local assistant interface."""
    template_path = Path(__file__).resolve().parent.parent / "web" / "browser_ai.html"
    html = template_path.read_text(encoding="utf-8")
    context = json.dumps(str(dataset_context or "")[:10000], ensure_ascii=False)
    context = context.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    palette = json.dumps(THEMES.get(theme, THEMES["Light"]))
    return html.replace("__CONTEXT__", context).replace("__PALETTE__", palette)
