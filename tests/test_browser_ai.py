from browser_ai import build_browser_ai_html


def test_browser_ai_template_is_fully_rendered():
    html = build_browser_ai_html("Revenue summary: 100", "Ocean")
    assert "__CONTEXT__" not in html
    assert "__PALETTE__" not in html
    assert "Qwen3.5-0.8B-q4f16_1-MLC" in html
    assert "Qwen3.5-2B-q4f16_1-MLC" in html
    assert "navigator.gpu" in html
    assert "stream:true" in html
    assert "Revenue summary: 100" in html
    assert "--accent" in html


def test_browser_ai_context_is_script_safe():
    html = build_browser_ai_html("</script><script>alert(1)</script>", "Light")
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html


def test_browser_ai_uses_fallback_theme_for_unknown_theme():
    html = build_browser_ai_html("", "not-a-theme")
    assert "#2563eb" in html
    assert 'const DATASET_CONTEXT = "";' in html


def test_browser_ai_disables_stale_model_after_load_failure_and_explains_fallback():
    html = build_browser_ai_html("", "Light")
    assert "activeModel = null" in html
    assert "Model not loaded" in html
    assert "Dataset tools (no AI server)" in html
    assert "loadButton.disabled = value || !supported" in html
