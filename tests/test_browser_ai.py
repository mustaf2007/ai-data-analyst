from browser_ai import build_browser_ai_html


def test_browser_ai_template_is_fully_rendered():
    html = build_browser_ai_html("Revenue summary: 100", "Ocean")
    assert "__CONTEXT__" not in html
    assert "__PALETTE__" not in html
    assert "Llama-3.2-1B-Instruct-q4f32_1-MLC" in html
    assert "Qwen3.5-0.8B-q4f16_1-MLC" in html
    assert "Qwen3.5-2B-q4f16_1-MLC" in html
    assert "navigator.gpu" in html
    assert 'adapter.features?.has("shader-f16")' in html
    assert "Selected model needs shader-f16" in html
    assert "Compatibility · Llama 3.2 · 1B" in html
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


def test_browser_ai_invalidates_disposed_engine_after_generation_failure():
    html = build_browser_ai_html("", "Light")
    assert "already been disposed|device lost|device was lost|device is lost" in html
    assert 'engine = null;' in html
    assert 'Model unavailable · reload required' in html
    assert 'Click Load model to initialise a fresh engine.' in html
