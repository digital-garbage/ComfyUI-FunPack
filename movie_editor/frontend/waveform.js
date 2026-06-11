/** Decode audio assets to peak arrays and paint timeline waveforms. */
window.FunPackWaveform = (function () {
  const _peaks = new Map();

  function _bucketCount(widthPx) {
    return Math.max(32, Math.min(512, Math.floor((widthPx || 200) / 2)));
  }

  async function peaksFromUrl(key, url, buckets) {
    const cacheKey = `${key}:${buckets}`;
    if (_peaks.has(cacheKey)) return _peaks.get(cacheKey);
    const job = (async () => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`fetch ${res.status}`);
      const buf = await res.arrayBuffer();
      const ctx = new AudioContext();
      try {
        const decoded = await ctx.decodeAudioData(buf.slice(0));
        const ch = decoded.getChannelData(0);
        const block = Math.max(1, Math.floor(ch.length / buckets));
        const peaks = new Float32Array(buckets);
        for (let i = 0; i < buckets; i++) {
          let max = 0;
          const start = i * block;
          for (let j = 0; j < block; j++) {
            max = Math.max(max, Math.abs(ch[start + j] || 0));
          }
          peaks[i] = max;
        }
        return { peaks, durationSec: decoded.duration };
      } finally {
        ctx.close();
      }
    })();
    _peaks.set(cacheKey, job);
    return job;
  }

  function _sizeCanvas(canvas) {
    const w = Math.max(1, canvas.clientWidth || canvas.offsetWidth || 100);
    const h = Math.max(1, canvas.clientHeight || canvas.offsetHeight || 28);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    return { w: canvas.width, h: canvas.height, dpr };
  }

  function paint(canvas, peaks, opts) {
    if (!canvas || !peaks || !peaks.length) return;
    const g = canvas.getContext("2d");
    if (!g) return;
    const { w, h, dpr } = _sizeCanvas(canvas);
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, w, h);
    const fill = (opts && opts.color) || "rgba(45, 212, 191, 0.55)";
    const mid = h / 2;
    const step = w / peaks.length;
    g.fillStyle = fill;
    for (let i = 0; i < peaks.length; i++) {
      const amp = peaks[i] * mid * 0.92;
      const x = i * step;
      g.fillRect(x, mid - amp, Math.max(1, step * 0.85), amp * 2);
    }
    if (dpr !== 1) g.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function paintPlaceholder(canvas, opts) {
    if (!canvas) return;
    const g = canvas.getContext("2d");
    if (!g) return;
    const { w, h } = _sizeCanvas(canvas);
    g.clearRect(0, 0, w, h);
    const fill = (opts && opts.color) || "rgba(45, 138, 106, 0.35)";
    g.fillStyle = fill;
    const mid = h / 2;
    const n = Math.max(24, Math.floor(w / 4));
    for (let i = 0; i < n; i++) {
      const t = i / n;
      const amp = (0.15 + 0.12 * Math.sin(t * 28) + 0.08 * Math.sin(t * 11)) * mid;
      g.fillRect(i * (w / n), mid - amp, Math.max(1, w / n * 0.7), amp * 2);
    }
  }

  function attach(canvas, key, url, opts) {
    if (!canvas || !url) {
      paintPlaceholder(canvas, opts);
      return;
    }
    const buckets = _bucketCount((opts && opts.width) || canvas.clientWidth || 200);
    peaksFromUrl(key, url, buckets).then((data) => {
      if (!canvas.isConnected) return;
      paint(canvas, data.peaks, opts);
      if (opts && typeof opts.onDuration === "function" && data.durationSec > 0) {
        opts.onDuration(data.durationSec);
      }
    }).catch(() => paintPlaceholder(canvas, opts));
  }

  return { attach, paint, paintPlaceholder, peaksFromUrl };
})();
