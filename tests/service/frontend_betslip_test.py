from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_betslip_renders_aggregate_and_pick_metrics():
    source = (FRONTEND / "features/betslip/BetslipPage.jsx").read_text(encoding="utf-8")
    for label in (
        "Quota combinata",
        "Quota void combinata",
        "Edge combinato",
        "Expected ROI",
        "Probabilità corretta",
        "Quota void modello",
        "Situazione",
        "Esito",
    ):
        assert label in source


def test_betslip_has_distinct_decision_and_settlement_badges():
    source = (FRONTEND / "features/betslip/BetslipPage.jsx").read_text(encoding="utf-8")
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    assert "leg.decision || leg.situation" in source
    assert 'leg.status === "VOID"' in source
    for css_class in ("value-play", "value-borderline", "value-no-bet", "value-unavailable"):
        assert css_class in styles


def test_opening_page_uses_read_only_get_requests():
    api = (FRONTEND / "api.js").read_text(encoding="utf-8")
    section = api.split("export function getBetslipGenerate", 1)[1].split("function normalizeJobBody", 1)[0]
    assert "method:" not in section
    assert "/betslip/generate" in section
    assert "/betslip/official" in section
    assert "/betslip/official/statistics" in section


def test_frontend_does_not_duplicate_betslip_formulas():
    source = (FRONTEND / "features/betslip/BetslipPage.jsx").read_text(encoding="utf-8")
    for formula in ("1 / p_model", "adjusted_probability * combined_odd", "market_odd - model_void_odd"):
        assert formula not in source


def test_betslip_uses_compact_coupon_structure():
    source = (FRONTEND / "features/betslip/BetslipPage.jsx").read_text(encoding="utf-8")
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    for label in (
        "Simula puntata",
        "Schedine",
        "Ufficiali",
        "Solo PLAY",
        "Prudenti",
        "Bilanciate",
        "Spinte",
        "Vincita potenziale",
        "Profitto potenziale",
        "Copia schedina",
        "Stampa / PDF",
    ):
        assert label in source
    for css_class in (
        ".betslip-toolbar",
        ".betslip-view-tabs",
        ".betslip-legend",
        ".betslip-list",
        ".slip-card-header",
        ".slip-footer",
    ):
        assert css_class in styles


def test_betslip_settlement_is_not_color_only():
    source = (FRONTEND / "features/betslip/BetslipPage.jsx").read_text(encoding="utf-8")
    for label in ("Vinta", "Persa", "In corso", "Rimborsata", "Proposta"):
        assert label in source
    assert "legStatusLabel(leg.status, slip.is_official)" in source
