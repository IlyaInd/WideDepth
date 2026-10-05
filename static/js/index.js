'use strict';

const figureDialog = document.getElementById('figure-dialog');
const expandedFigure = document.getElementById('expanded-figure');
const figureTitle = document.getElementById('figure-title');
const originalImage = document.getElementById('original-image');
let figureTrigger;

// Links still open the original image when JavaScript or dialogs are unavailable.
if (typeof figureDialog.showModal === 'function') {
  document.querySelectorAll('.figure-link').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
      event.preventDefault();
      figureTrigger = link;
      const thumbnail = link.querySelector('img');
      expandedFigure.src = link.href;
      expandedFigure.alt = thumbnail.alt;
      figureTitle.textContent = thumbnail.alt;
      originalImage.href = link.href;
      figureDialog.showModal();
    });
  });
  document.getElementById('close-figure').addEventListener('click', () => figureDialog.close());
  figureDialog.addEventListener('click', event => {
    const bounds = figureDialog.getBoundingClientRect();
    if (event.target === figureDialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) {
      figureDialog.close();
    }
  });
  figureDialog.addEventListener('close', () => figureTrigger?.focus({ preventScroll: true }));
}

const copyButton = document.getElementById('copy-citation');
const copyStatus = document.getElementById('copy-status');
const citation = document.getElementById('bibtex-code');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyStatus.textContent = 'BibTeX copied to clipboard.';
  } catch {
    // Select the citation for manual copying when clipboard permission is denied.
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});

const scrollButton = document.querySelector('.scroll-to-top');
const updateScrollButton = () => { scrollButton.hidden = window.scrollY < 500; };
window.addEventListener('scroll', updateScrollButton, { passive: true });
updateScrollButton();
scrollButton.addEventListener('click', () => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  window.scrollTo({ top: 0, behavior: reducedMotion ? 'instant' : 'smooth' });
});
