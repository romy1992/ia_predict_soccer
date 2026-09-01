function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function clickMenuButton(ui, label) {
  const snapshot = await ui.snapshot();
  const pattern = new RegExp(`@(e\\d+) button "${escapeRegex(label)}"`);
  const match = snapshot.match(pattern);
  if (!match) {
    return false;
  }

  await ui.click(`@${match[1]}`);
  return true;
}

export default async function run(page, ui) {
  const imageDir = "C:/Users/trott/git/ia_predict_soccer/docs/images";

  await page.setViewportSize({ width: 1500, height: 920 });
  await page.waitForTimeout(1500);

  await page.screenshot({
    path: `${imageDir}/frontend-dashboard.png`,
    fullPage: true,
  });

  const sections = [
    { label: "Partite in diretta", file: "frontend-live.png" },
    { label: "Partite del giorno", file: "frontend-today.png" },
    { label: "Storico previsioni", file: "frontend-predictions.png" },
    { label: "Operazioni ML", file: "frontend-ops-ml.png" },
  ];

  const generated = ["frontend-dashboard.png"];

  for (const section of sections) {
    const opened = await clickMenuButton(ui, section.label);
    if (!opened) {
      continue;
    }
    await page.waitForTimeout(1000);
    await page.screenshot({
      path: `${imageDir}/${section.file}`,
      fullPage: true,
    });
    generated.push(section.file);
  }

  const finalSnapshot = await ui.snapshot();
  const detailButtonMatch = finalSnapshot.match(/@(e\d+) button "Apri"/);
  if (detailButtonMatch) {
    await ui.click(`@${detailButtonMatch[1]}`);
    await page.waitForTimeout(1200);
    await page.screenshot({
      path: `${imageDir}/frontend-match-detail.png`,
      fullPage: true,
    });
    generated.push("frontend-match-detail.png");
  }

  return { generated };
}

