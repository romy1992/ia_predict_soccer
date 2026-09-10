from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_match_center_contains_required_columns_and_distinct_void_concepts():
    source = (FRONTEND / "features/matches/components/MatchTable.jsx").read_text(encoding="utf-8")
    for label in (
        "Pronostico",
        "Probabilità IA",
        "Quota mercato",
        "Quota void IA",
        "Edge",
        "ROI atteso",
        "Situazione",
        "Esito ufficiale",
    ):
        assert label in source
    assert "stato VOID" in source
    assert "Quota di pareggio economico" in source


def test_situation_badges_and_no_bet_filter_are_present():
    page = (FRONTEND / "features/pages/DashboardPage.jsx").read_text(encoding="utf-8")
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    for label in ("PLAY", "BORDERLINE", "NO BET", "SENZA QUOTA"):
        assert label in page
    assert "value-play" in styles
    assert "value-borderline" in styles
    assert "value-no-bet" in styles
    assert "value-unavailable" in styles


def test_market_filter_selects_its_own_backend_decision_card():
    formatter = (FRONTEND / "features/shared/formatters.js").read_text(encoding="utf-8")
    assert "card.market === market" in formatter
    assert "card.is_market_best" in formatter


def test_mobile_view_keeps_void_odd_and_situation_visible():
    table = (FRONTEND / "features/matches/components/MatchTable.jsx").read_text(encoding="utf-8")
    mobile = table.split('className="match-center-mobile"', maxsplit=1)[1]
    assert "Quota void IA" in mobile
    assert "Situazione" in mobile
    assert "ROI atteso" in mobile


def test_frontend_only_formats_backend_betting_metrics():
    frontend_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in FRONTEND.rglob("*")
        if path.suffix in {".js", ".jsx"}
    )
    forbidden_formulas = (
        "1 / predicted_probability",
        "1 / p_model",
        "market_odd / model_void_odd",
        "predicted_probability * market_odd",
    )
    assert not any(formula in frontend_source for formula in forbidden_formulas)
