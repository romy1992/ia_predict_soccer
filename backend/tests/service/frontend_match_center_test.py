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
        "Giocata ufficiale",
    ):
        assert label in source
    assert "stato VOID" in source
    assert "Quota di pareggio economico" in source
    assert "Non registrata" in source
    assert "Gialli effettivi" in source
    assert "Rossi effettivi" in source
    assert "Totale corners" in source


def test_search_button_is_explicitly_scoped_to_match_text():
    source = (FRONTEND / "features/layout/TopFilters.jsx").read_text(encoding="utf-8")
    assert source.count("Cerca match") >= 2
    assert "Data e mercato si aggiornano automaticamente" in source
    assert 'className="filter-card filter-search"' in source
    assert 'className="filter-card filter-refresh"' in source
    assert "Aggiornamento dati" in source


def test_top_filters_have_distinct_responsive_spaces():
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    assert ".filter-card {" in styles
    assert ".filter-search-controls {" in styles
    assert ".filter-refresh {" in styles
    assert "grid-column: 1 / -1" in styles


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


def test_all_markets_keeps_prediction_chips_and_winning_pick_metrics():
    table = (FRONTEND / "features/matches/components/MatchTable.jsx").read_text(encoding="utf-8")
    badges = (FRONTEND / "features/matches/components/PredictionBadges.jsx").read_text(encoding="utf-8")
    assert "showAllMarkets" in table
    assert "Tutti i mercati" in table
    assert "Pronostico vincitore" in table
    assert "<PredictionBadges" in table
    assert "prediction-value-label" in badges
    assert "decision?.value_label" in badges


def test_single_market_keeps_analytic_view():
    table = (FRONTEND / "features/matches/components/MatchTable.jsx").read_text(encoding="utf-8")
    assert 'showAllMarkets ? "Pronostico vincitore" : "Pronostico"' in table
    for backend_field in (
        "predicted_probability",
        "market_odd",
        "model_void_odd",
        "odds_edge_absolute",
        "expected_roi_percent",
    ):
        assert backend_field in table


def test_dashboard_uses_one_bundle_request_and_client_side_market_filters():
    app = (FRONTEND / "App.jsx").read_text(encoding="utf-8")
    api = (FRONTEND / "api.js").read_text(encoding="utf-8")
    load_dashboard = app.split("const loadDashboardData", maxsplit=1)[1].split(
        "const handleForceRefreshDay", maxsplit=1
    )[0]

    assert "getDashboardBundle" in load_dashboard
    assert "getDashboardOverview" not in load_dashboard
    assert "getDashboardLive" not in load_dashboard
    assert "getDashboardDay" not in load_dashboard
    assert 'request(`/dashboard/bundle?' in api
    assert "initialLoadStartedRef.current" in app
    assert 'activePage !== "dashboard"' in app
    assert "filterRowByMarket(row, selectedMarket)" in app
