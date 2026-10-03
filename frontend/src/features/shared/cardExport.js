import html2canvas from "html2canvas";
import JSZip from "jszip";

// Esclude dallo screenshot i controlli dell'interfaccia (bottoni azione) e i
// dettagli tecnici collassati: il marker nativo di <summary> non viene
// renderizzato correttamente da html2canvas (compare come "1." invece del
// triangolo), quindi l'intera sezione chiusa va esclusa dall'immagine.
const EXPORT_IGNORE_SELECTORS = [".slip-actions", ".slip-technical"];

async function renderCardToBlob(node) {
  const canvas = await html2canvas(node, {
    backgroundColor: "#ffffff",
    scale: Math.min(window.devicePixelRatio || 1, 2),
    ignoreElements: (el) => EXPORT_IGNORE_SELECTORS.some((selector) => el.matches?.(selector)),
  });
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export async function downloadCardAsImage(node, filename) {
  const blob = await renderCardToBlob(node);
  triggerDownload(blob, filename);
}

export async function downloadCardsAsZip(entries, zipFilename) {
  const zip = new JSZip();
  for (const { node, filename } of entries) {
    const blob = await renderCardToBlob(node);
    zip.file(filename, blob);
  }
  const zipBlob = await zip.generateAsync({ type: "blob" });
  triggerDownload(zipBlob, zipFilename);
}
