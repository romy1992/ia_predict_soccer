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
    section = api.split("export function getBetslipGenerate", 1)[1].split(
        "export function saveBetslipGeneration",
        1,
    )[0]
    assert "method:" not in section
    assert "/betslip/generate" in section
    assert 'request(`/betslip/generate?${params.toString()}`)' in section


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
        "Consigliate",
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
    assert "legStatusLabel(leg.status, slip.is_official, hasShadowStatus)" in source


def test_betslip_unifies_daily_market_and_slip_statistics():
    source = (FRONTEND / "features/betslip/BetslipPage.jsx").read_text(encoding="utf-8")
    for label in (
        "Statistiche Betting",
        "Panoramica",
        "Mercati",
        "Schedine",
        "Attività generata",
        "Performance schedine ufficiali",
        "Proposte salvate",
        "Revisioni",
    ):
        assert label in source
    assert "report?.markets?.daily" in source
    assert "report?.official_daily" in source


def test_manual_generation_is_explicitly_persisted_but_page_load_remains_read_only():
    api = (FRONTEND / "api.js").read_text(encoding="utf-8")
    assert 'request(`/betslip/generate?${params.toString()}`)' in api
    assert "/betslip/generate/snapshot" in api
    assert 'method: "POST"' in api.split("export function saveBetslipGeneration", 1)[1].split(
        "export function getOfficialBetslips",
        1,
    )[0]


def test_past_dates_are_read_only_and_load_saved_proposals():
    page = (FRONTEND / "features/betslip/BetslipPage.jsx").read_text(encoding="utf-8")
    app = (FRONTEND / "App.jsx").read_text(encoding="utf-8")
    api = (FRONTEND / "api.js").read_text(encoding="utf-8")

    assert "Solo consultazione" in page
    assert "senza rigenerazioni retroattive" in page
    assert "disabled={isLoading || isPastDate}" in page
    assert "getSavedBetslipProposals" in app
    assert "isPastDate" in app
    proposal_section = api.split("export function getSavedBetslipProposals", 1)[1].split(
        "export function getOfficialBetslips",
        1,
    )[0]
    assert "/betslip/proposals" in proposal_section
    assert "method:" not in proposal_section


def test_betslip_exposes_separate_decision_tabs_and_shadow_portfolios():
    source = (FRONTEND / "features/betslip/BetslipPage.jsx").read_text(encoding="utf-8")
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    for label in (
        "Consigliate",
        "Sperimentali",
        "Non consigliate",
        "Portafogli simulati",
        "Complessivo simulato",
        "PLAY simulato",
        "BORDERLINE simulato",
        "NO BET simulato",
        "ROI simulato",
    ):
        assert label in source
    assert "report?.decision_groups" in source
    assert ".betslip-decision-tabs" in styles
    assert ".shadow-portfolios" in styles
