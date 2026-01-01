async function copyText(anchorEl, textToCopy) {
  try {
    // Prefer the modern Clipboard API
    try {
      await navigator.clipboard.writeText(textToCopy);
    } catch (clipboardErr) {
      // Fallback: use a temporary textarea for execCommand('copy')
      const ta = document.createElement('textarea');
      ta.value = textToCopy;
      ta.style.position = 'fixed';
      ta.style.top = '0';
      ta.style.left = '0';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      if (!ok) {
        throw clipboardErr;
      }
    }

    // Visual feedback
    const originalTitle = anchorEl.getAttribute('title');
    anchorEl.setAttribute('title', 'copied!');
    const img = anchorEl.querySelector('img');
    if (img) img.style.opacity = '0.5';
    setTimeout(() => {
      anchorEl.setAttribute('title', originalTitle || 'copy');
      if (img) img.style.opacity = '1';
    }, 250);

  } catch (err) {
    console.error('Failed to copy text: ', err);
    const originalTitle = anchorEl.getAttribute('title');
    anchorEl.setAttribute('title', 'Failed to copy!');
    setTimeout(() => {
      anchorEl.setAttribute('title', originalTitle || 'copy');
    }, 2000);
  }
}