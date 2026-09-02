import DashboardFeaturePage from "../dashboard/DashboardFeaturePage";
import DataCenterPage from "../data-center/DataCenterPage";
import DataQualityPage from "../data-quality/DataQualityPage";
import MatchesDayPage from "../matches/MatchesDayPage";
import MatchesLivePage from "../matches/MatchesLivePage";
import MlLabPage from "../ml-lab/MlLabPage";
import OraclePage from "../oracle/OraclePage";

export default function AppRouter({ activePage, props }) {
  if (activePage === "dashboard") {
    return <DashboardFeaturePage {...props.dashboard} />;
  }
  if (activePage === "live") {
    return <MatchesLivePage {...props.live} />;
  }
  if (activePage === "today") {
    return <MatchesDayPage {...props.today} />;
  }
  if (activePage === "predictions") {
    return <OraclePage {...props.predictions} />;
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

  return <DashboardFeaturePage {...props.dashboard} />;
}




