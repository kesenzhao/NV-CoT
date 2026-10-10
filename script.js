const header = document.querySelector("[data-header]");

const updateHeader = () => {
  header?.classList.toggle("scrolled", window.scrollY > 24);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));

const benchmarkData = {
  vstar: { base: 86.4, ours: 89.0, note: "Single-call RL setting · DeepEyes-7B backbone" },
  hr4k: { base: 73.5, ours: 74.8, note: "Single-call RL setting · DeepEyes-7B backbone" },
  hr8k: { base: 70.4, ours: 72.0, note: "Single-call RL setting · DeepEyes-7B backbone" },
};

const tabs = document.querySelectorAll("[data-benchmark]");
const baseBar = document.querySelector("[data-base-bar]");
const oursBar = document.querySelector("[data-ours-bar]");
const baseValue = document.querySelector("[data-base-value]");
const oursValue = document.querySelector("[data-ours-value]");
const chartNote = document.querySelector("[data-chart-note]");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const next = benchmarkData[tab.dataset.benchmark];
    tabs.forEach((item) => {
      const active = item === tab;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    baseBar.style.width = `${next.base}%`;
    oursBar.style.width = `${next.ours}%`;
    baseValue.textContent = next.base.toFixed(1);
    oursValue.textContent = next.ours.toFixed(1);
    chartNote.textContent = next.note;
  });
});

const copyButton = document.querySelector("[data-copy-citation]");
const citation = document.querySelector("[data-citation]");

copyButton?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.textContent = "Copied ✓";
    window.setTimeout(() => (copyButton.textContent = "Copy citation"), 1800);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = "Selected";
  }
});
