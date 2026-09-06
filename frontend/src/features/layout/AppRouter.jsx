import DashboardFeaturePage from "../dashboard/DashboardFeaturePage";
import DataCenterPage from "../data-center/DataCenterPage";
import DataQualityPage from "../data-quality/DataQualityPage";
import BetslipPage from "../betslip/BetslipPage";
import MlLabPage from "../ml-lab/MlLabPage";
import MonitoringPage from "../monitoring/MonitoringPage";
import OraclePage from "../oracle/OraclePage";
import OracleMatchDetailPage from "../oracle/OracleMatchDetailPage";
import SettingsPage from "../settings/SettingsPage";

export default function AppRouter({ activePage, props }) {
  if (activePage === "dashboard") {
    return <DashboardFeaturePage {...props.dashboard} />;
  }
  if (activePage === "predictions") {
    return <OraclePage {...props.predictions} />;
  }
  if (activePage === "oracle-detail") {
    return <OracleMatchDetailPage {...props.oracleDetail} />;
  }
  if (activePage === "betslip") {
    return <BetslipPage {...props.betslip} />;
  }
  if (activePage === "data-center" || activePage === "ops") {
    return <DataCenterPage {...props.dataCenter} />;
  }
  if (activePage === "data-quality") {
    return <DataQualityPage {...props.dataQuality} />;
  }
  if (activePage === "ml-lab") {
    return <MlLabPage {...props.mlLab} />;
  }
  if (activePage === "monitoring") {
    return <MonitoringPage {...props.monitoring} />;
  }
  if (activePage === "settings") {
    return <SettingsPage {...props.settings} />;
  }

  return <DashboardFeaturePage {...props.dashboard} />;
}



