/* ======================================================================
   Interface: drawing, controls, verification panel
   ====================================================================== */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var canvas = $("plot"), ctx = canvas.getContext("2d");
  var wrap = $("plotWrap");

  var FIELD_LABEL = {
    speed: "Speed / U_lid (dimensionless)",
    u: "u / U_lid (dimensionless)",
    v: "v / U_lid (dimensionless)",
    p: "p / (\u03C1 U_lid\u00B2) (dimensionless)"
  };
  var FIELD_NAME = { speed: "Speed", u: "u", v: "v", p: "Pressure" };

  var state = {
    model: null, net: null, N: 50,
    Re: 100, field: "speed", streamlines: true, smooth: false, lock: false,
    data: null, version: 0, slCache: null, sweep: null,
    hover: null, layout: null
  };

  /* ---------- colour maps ---------- */
  function hex(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
  function makeLUT(stops) {
    var cols = stops.map(hex), lut = new Uint8ClampedArray(256 * 3);
    for (var k = 0; k < 256; k++) {
      var t = k / 255 * (cols.length - 1), i = Math.min(cols.length - 2, Math.floor(t)), f = t - i;
      for (var c = 0; c < 3; c++) lut[k * 3 + c] = cols[i][c] + (cols[i + 1][c] - cols[i][c]) * f;
    }
    return lut;
  }
  var LUT_SEQ = makeLUT(["#440154", "#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"]);
  var LUT_DIV = makeLUT(["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"]);

  /* ---------- value ranges ---------- */
  function fieldRange(field, arr) {
    var mx = 0, i;
    if (field === "speed") {
      for (i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i];
      return { lo: 0, hi: mx > 0 ? mx : 1, div: false };
    }
    for (i = 0; i < arr.length; i++) { var a = Math.abs(arr[i]); if (a > mx) mx = a; }
    if (!(mx > 0)) mx = 1;
    return { lo: -mx, hi: mx, div: true };
  }
  // Largest value of each field over a coarse Re sweep, used when the colour scale is kept fixed.
  function sweepRanges() {
    if (state.sweep) return state.sweep;
    var net = state.net, tmp = CavityNN.newFields(state.N), best = { speed: 0, u: 0, v: 0, p: 0 };
    var n = 29;
    for (var k = 0; k < n; k++) {
      var Re = net.reMin + (net.reMax - net.reMin) * k / (n - 1);
      CavityNN.evalGrid(net, Re, tmp);
      ["speed", "u", "v", "p"].forEach(function (f) {
        var arr = tmp[f];
        for (var i = 0; i < arr.length; i++) { var a = Math.abs(arr[i]); if (a > best[f]) best[f] = a; }
      });
    }
    state.sweep = best;
    return best;
  }
  function currentRange() {
    var f = state.field, r = fieldRange(f, state.data[f]);
    if (state.lock) {
      var m = sweepRanges()[f] || 1;
      r = (f === "speed") ? { lo: 0, hi: m, div: false } : { lo: -m, hi: m, div: true };
    }
    return r;
  }

  function niceTicks(lo, hi, target) {
    var span = hi - lo, raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag, step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
    var ticks = [], t = Math.ceil(lo / step - 1e-9) * step;
    for (; t <= hi + step * 1e-6; t += step) ticks.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    var dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    return { ticks: ticks, dec: Math.min(dec, 6) };
  }

  /* ---------- streamlines ---------- */
  function traceStreamlines(u, v, N) {
    var G = 90, occ = new Int32Array(G * G).fill(-1), lines = [], id = 0;
    var lo = 0.5 / N, span = N - 1;
    function vel(x, y, out) {
      var fx = Math.min(span, Math.max(0, x * N - 0.5)), fy = Math.min(span, Math.max(0, y * N - 0.5));
      var i0 = Math.min(N - 2, Math.floor(fx)), j0 = Math.min(N - 2, Math.floor(fy));
      var tx = fx - i0, ty = fy - j0, a = j0 * N + i0, b = a + 1, c = a + N, d = c + 1;
      out[0] = (u[a] * (1 - tx) + u[b] * tx) * (1 - ty) + (u[c] * (1 - tx) + u[d] * tx) * ty;
      out[1] = (v[a] * (1 - tx) + v[b] * tx) * (1 - ty) + (v[c] * (1 - tx) + v[d] * tx) * ty;
    }
    var vv = [0, 0];
    function dirAt(x, y, sgn, out) {
      vel(x, y, vv);
      var s = Math.hypot(vv[0], vv[1]);
      if (s < 1e-5) return false;
      out[0] = sgn * vv[0] / s; out[1] = sgn * vv[1] / s;
      return true;
    }
    function cellOf(x, y) {
      return Math.min(G - 1, Math.floor(y * G)) * G + Math.min(G - 1, Math.floor(x * G));
    }
    var ds = 0.004, k1 = [0, 0], k2 = [0, 0], k3 = [0, 0], k4 = [0, 0];
    function march(x, y, sgn, lid) {
      var pts = [];
      for (var s = 0; s < 800; s++) {
        if (!dirAt(x, y, sgn, k1)) break;
        if (!dirAt(x + 0.5 * ds * k1[0], y + 0.5 * ds * k1[1], sgn, k2)) break;
        if (!dirAt(x + 0.5 * ds * k2[0], y + 0.5 * ds * k2[1], sgn, k3)) break;
        if (!dirAt(x + ds * k3[0], y + ds * k3[1], sgn, k4)) break;
        x += ds / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        y += ds / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        if (x < 0 || x > 1 || y < 0 || y > 1) break;
        var c = cellOf(x, y);
        if (occ[c] !== -1 && occ[c] !== lid) break;
        occ[c] = lid;
        pts.push([x, y]);
      }
      return pts;
    }
    var S = 24;
    for (var sj = 0; sj < S; sj++) {
      for (var si = 0; si < S; si++) {
        var x0 = (si + 0.5) / S, y0 = (sj + 0.5) / S, c0 = cellOf(x0, y0);
        if (occ[c0] !== -1) continue;
        id++; occ[c0] = id;
        var fwd = march(x0, y0, 1, id), bwd = march(x0, y0, -1, id);
        if (fwd.length + bwd.length < 10) continue;
        bwd.reverse();
        lines.push(bwd.concat([[x0, y0]], fwd));
      }
    }
    return lines;
  }

  /* ---------- drawing ---------- */
  var offscreen = document.createElement("canvas");

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function draw() {
    if (!state.data) return;
    var dpr = window.devicePixelRatio || 1;
    var cssW = Math.max(260, wrap.clientWidth);
    var narrow = cssW < 480;
    var m = narrow ? { l: 50, r: 74, t: 54, b: 50 } : { l: 62, r: 92, t: 58, b: 56 };
    var plot = cssW - m.l - m.r, cssH = plot + m.t + m.b;
    canvas.style.height = cssH + "px";
    var pw = Math.round(cssW * dpr), ph = Math.round(cssH * dpr);
    if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    state.layout = { l: m.l, t: m.t, plot: plot };

    var N = state.N, field = state.field, arr = state.data[field];
    var ink = css("--ink"), muted = css("--muted"), line = css("--line");
    var range = currentRange(), lut = range.div ? LUT_DIV : LUT_SEQ;
    var span = range.hi - range.lo;

    // Heatmap pixels. Grid row j (y increasing) goes to image row N-1-j, so y points upward.
    offscreen.width = N; offscreen.height = N;
    var octx = offscreen.getContext("2d"), img = octx.createImageData(N, N), px = img.data;
    for (var j = 0; j < N; j++) {
      var row = N - 1 - j;
      for (var i = 0; i < N; i++) {
        var t = (arr[j * N + i] - range.lo) / span;
        var q = Math.max(0, Math.min(255, Math.round(t * 255))) * 3, o = (row * N + i) * 4;
        px[o] = lut[q]; px[o + 1] = lut[q + 1]; px[o + 2] = lut[q + 2]; px[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = state.smooth;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(offscreen, m.l, m.t, plot, plot);

    // Streamlines
    if (state.streamlines) {
      if (!state.slCache || state.slCache.version !== state.version) {
        state.slCache = { version: state.version, lines: traceStreamlines(state.data.u, state.data.v, N) };
      }
      ctx.save();
      ctx.beginPath(); ctx.rect(m.l, m.t, plot, plot); ctx.clip();
      ctx.lineWidth = 1.1; ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.strokeStyle = range.div ? "rgba(23,35,58,0.7)" : "rgba(255,255,255,0.8)";
      state.slCache.lines.forEach(function (pl) {
        ctx.beginPath();
        for (var k = 0; k < pl.length; k++) {
          var X = m.l + pl[k][0] * plot, Y = m.t + (1 - pl[k][1]) * plot;
          if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        }
        ctx.stroke();
      });
      ctx.restore();
    }

    // Hover marker
    if (state.hover) {
      var hx = m.l + (state.hover.i + 0.5) / N * plot, hy = m.t + (1 - (state.hover.j + 0.5) / N) * plot;
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.beginPath(); ctx.arc(hx, hy, 6, 0, 6.2832); ctx.stroke();
      ctx.lineWidth = 1.5; ctx.strokeStyle = "#ffffff";
      ctx.beginPath(); ctx.arc(hx, hy, 6, 0, 6.2832); ctx.stroke();
    }

    // Frame and axes
    ctx.strokeStyle = ink; ctx.lineWidth = 1;
    ctx.strokeRect(m.l - 0.5, m.t - 0.5, plot + 1, plot + 1);
    ctx.fillStyle = ink; ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (var a = 0; a <= 5; a++) {
      var v = a / 5, X2 = m.l + v * plot;
      ctx.beginPath(); ctx.moveTo(X2, m.t + plot); ctx.lineTo(X2, m.t + plot + 5); ctx.stroke();
      ctx.fillText(v.toFixed(1), X2, m.t + plot + 7);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (var b = 0; b <= 5; b++) {
      var w = b / 5, Y2 = m.t + (1 - w) * plot;
      ctx.beginPath(); ctx.moveTo(m.l - 5, Y2); ctx.lineTo(m.l, Y2); ctx.stroke();
      ctx.fillText(w.toFixed(1), m.l - 8, Y2);
    }
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillText("x/L", m.l + plot / 2, cssH - 10);
    ctx.save(); ctx.translate(13, m.t + plot / 2); ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle"; ctx.fillText("y/L", 0, 0); ctx.restore();

    // Title block
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = ink; ctx.font = "600 16px system-ui, sans-serif";
    ctx.fillText("Neural-network prediction", m.l, 22);
    ctx.fillStyle = muted; ctx.font = "12.5px system-ui, sans-serif";
    ctx.fillText(FIELD_NAME[field] + " at Re = " + state.Re.toFixed(2) + ", " + N + " \u00D7 " + N + " cell centres" + (state.Re > state.net.trainMax ? ", extrapolating beyond Re " + state.net.trainMax : ""), m.l, 40);
    // Lid direction cue on the top edge
    ctx.textAlign = "right"; ctx.fillStyle = muted;
    ctx.fillText("moving lid \u2192", m.l + plot, m.t - 8);

    // Colourbar
    var cbx = m.l + plot + (narrow ? 14 : 18), cbw = 14;
    var grad = ctx.createLinearGradient(0, m.t + plot, 0, m.t);
    for (var s = 0; s <= 20; s++) {
      var q2 = Math.round(s / 20 * 255) * 3;
      grad.addColorStop(s / 20, "rgb(" + lut[q2] + "," + lut[q2 + 1] + "," + lut[q2 + 2] + ")");
    }
    ctx.fillStyle = grad; ctx.fillRect(cbx, m.t, cbw, plot);
    ctx.strokeStyle = ink; ctx.strokeRect(cbx - 0.5, m.t - 0.5, cbw + 1, plot + 1);
    var nt = niceTicks(range.lo, range.hi, 5);
    ctx.fillStyle = ink; ctx.font = "11.5px system-ui, sans-serif";
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.strokeStyle = ink;
    nt.ticks.forEach(function (tv) {
      var Y3 = m.t + (1 - (tv - range.lo) / span) * plot;
      ctx.beginPath(); ctx.moveTo(cbx + cbw, Y3); ctx.lineTo(cbx + cbw + 4, Y3); ctx.stroke();
      ctx.fillText(tv.toFixed(nt.dec), cbx + cbw + 7, Y3);
    });
    ctx.save();
    ctx.translate(cssW - 8, m.t + plot / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = (narrow ? "11px" : "12.5px") + " system-ui, sans-serif"; ctx.fillStyle = ink;
    ctx.fillText(FIELD_LABEL[field], 0, 0);
    ctx.restore();
  }

  var rafPending = false;
  function scheduleRender() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(function () { rafPending = false; evaluateAndDraw(); });
  }

  function evaluateAndDraw() {
    var t0 = performance.now();
    CavityNN.evalGrid(state.net, state.Re, state.data);
    state.version++;
    var t1 = performance.now();
    draw();
    $("timing").textContent = (state.N * state.N).toLocaleString("en-US") + " network evaluations in " +
      (t1 - t0).toFixed(1) + " ms.";
    updateReadout();
  }

  /* ---------- readout under the pointer ---------- */
  function updateReadout() {
    var el = $("readout");
    if (!state.hover) { el.textContent = "Move the pointer over the heatmap."; return; }
    var h = state.hover, N = state.N, k = h.j * N + h.i, d = state.data;
    function f(x) { return (x < 0 ? "\u2212" : "") + Math.abs(x).toFixed(4); }
    el.innerHTML = '<div class="grid">' +
      "<span>x/L, y/L</span><b>" + ((h.i + 0.5) / N).toFixed(3) + ", " + ((h.j + 0.5) / N).toFixed(3) + "</b>" +
      "<span>Speed</span><b>" + f(d.speed[k]) + "</b>" +
      "<span>u</span><b>" + f(d.u[k]) + "</b>" +
      "<span>v</span><b>" + f(d.v[k]) + "</b>" +
      "<span>p</span><b>" + f(d.p[k]) + "</b></div>";
  }

  function pointerToCell(ev) {
    var L = state.layout; if (!L) return null;
    var r = canvas.getBoundingClientRect();
    var px = (ev.clientX - r.left - L.l) / L.plot, py = (ev.clientY - r.top - L.t) / L.plot;
    if (px < 0 || px >= 1 || py < 0 || py >= 1) return null;
    var N = state.N;
    return { i: Math.min(N - 1, Math.floor(px * N)), j: Math.min(N - 1, Math.floor((1 - py) * N)) };
  }
  function onPointer(ev) {
    var c = pointerToCell(ev);
    if (!c && !state.hover) return;
    if (c && state.hover && c.i === state.hover.i && c.j === state.hover.j) return;
    state.hover = c;
    draw(); updateReadout();
  }
  canvas.addEventListener("pointermove", onPointer);
  canvas.addEventListener("pointerdown", onPointer);
  canvas.addEventListener("pointerleave", function () { state.hover = null; draw(); updateReadout(); });

  /* ---------- Reynolds number controls ---------- */
  var reRange = $("reRange"), reNumber = $("reNumber");
  function setRe(val, fromNumberTyping) {
    var lo = state.net.reMin, hi = state.net.reMax;
    if (!isFinite(val)) return;
    var clamped = Math.min(hi, Math.max(lo, val));
    state.Re = clamped;
    reRange.value = clamped;
    if (!fromNumberTyping || clamped !== val) reNumber.value = String(Math.round(clamped * 100) / 100);
    scheduleRender();
  }
  reRange.addEventListener("input", function () { setRe(parseFloat(reRange.value), false); });
  reNumber.addEventListener("input", function () {
    var v = parseFloat(reNumber.value);
    if (isFinite(v) && v >= state.net.reMin && v <= state.net.reMax) setRe(v, true);
  });
  reNumber.addEventListener("change", function () {
    var v = parseFloat(reNumber.value);
    setRe(isFinite(v) ? v : state.Re, false);
  });
  $("fieldSel").addEventListener("change", function (e) { state.field = e.target.value; draw(); updateReadout(); drawCompare(); });
  $("streamChk").addEventListener("change", function (e) { state.streamlines = e.target.checked; draw(); });
  $("smoothChk").addEventListener("change", function (e) { state.smooth = e.target.checked; draw(); });
  $("lockChk").addEventListener("change", function (e) { state.lock = e.target.checked; draw(); });

  function buildChips() {
    var box = $("chips"); box.innerHTML = "";
    var lo = state.net.reMin, hi = state.net.reMax;
    [lo, 100, 300, 500, 1000].forEach(function (v) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = String(v);
      b.addEventListener("click", function () { setRe(v, false); });
      box.appendChild(b);
    });
  }

  /* ---------- verification panel ---------- */
  function fmtErr(e) { return e === 0 ? "0" : e.toExponential(2); }
  function runVerification() {
    var res = CavityNN.verify(state.net, state.model);
    var sum = $("verifySummary"), table = $("verifyTable");
    table.innerHTML = "";
    if (!res.available) {
      sum.innerHTML = '<span class="badge fail">No reference data</span><p class="small muted">The file has no verification inputs and outputs, so the check could not run.</p>';
      return res;
    }
    sum.innerHTML = '<span class="badge ' + (res.pass ? "pass" : "fail") + '">' + (res.pass ? "Pass" : "Fail") + "</span>" +
      '<p class="small" style="margin:8px 0 0">Largest difference between this page and the Python outputs over ' + res.rows.length +
      " points: <b>" + fmtErr(res.maxErr) + "</b> (allowed: " + res.tol.toExponential(0) + ").</p>";
    var head = "<tr><th>x</th><th>y</th><th>Re</th><th>max |diff|</th><th></th></tr>";
    table.innerHTML = head + res.rows.map(function (r) {
      return "<tr><td>" + r.x + "</td><td>" + r.y + "</td><td>" + r.Re + "</td><td>" + fmtErr(r.err) +
        '</td><td class="' + (r.ok ? "ok" : "no") + '">' + (r.ok ? "ok" : "too large") + "</td></tr>";
    }).join("");
    return res;
  }
  $("rerunBtn").addEventListener("click", function () {
    var r = runVerification();
    setMessage(r.available ? (r.pass ? "Check finished: all points within tolerance." : "Check finished: some points exceed the tolerance.") : "No reference data to check.", !r.pass);
  });

  function setMessage(text, isError) {
    var m = $("message"); m.textContent = text; m.className = "small " + (isError ? "error" : "muted");
  }

  /* ---------- loading a model ---------- */
  function useModel(model, label) {
    var net = CavityNN.compile(model); net.trainMax = net.reMax; net.reMax = 1000;          // throws a readable error if the file is unusable
    state.model = model; state.net = net; state.N = net.N;
    state.data = CavityNN.newFields(net.N);
    state.sweep = null; state.slCache = null; state.hover = null;
    reRange.min = net.reMin; reRange.max = net.reMax;
    reNumber.min = net.reMin; reNumber.max = net.reMax;
    $("reMinLabel").textContent = "Re " + net.reMin; $("reMaxLabel").textContent = net.reMax;
    state.Re = Math.min(net.reMax, Math.max(net.reMin, state.Re));
    reRange.value = state.Re; reNumber.value = String(Math.round(state.Re * 100) / 100);
    buildChips();
    var hidden = (model.settings && model.settings.hidden_nodes) ? model.settings.hidden_nodes : "?";
    var nl = model.layers.filter(function (l) { return l.type === "Linear"; }).length;
    $("modelInfo").textContent = "Model: " + label + ", " + (model.format || "unknown format") + ", " + nl +
      " linear layers (" + hidden + " hidden nodes).";
    runVerification();
    evaluateAndDraw(); buildCmp(); drawCompare();
  }

  $("fileInput").addEventListener("change", function (e) {
    var file = e.target.files && e.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      try {
        useModel(JSON.parse(reader.result), file.name);
        setMessage("Loaded " + file.name + ".", false);
      } catch (err) {
        setMessage("Could not use " + file.name + ": " + err.message, true);
      }
    };
    reader.onerror = function () { setMessage("Could not read " + file.name + ".", true); };
    reader.readAsText(file);
    e.target.value = "";
  });


  /* ---------- CFD vs network comparison ---------- */
  var CFD = window.CAVITY_CFD, cmpIdx = 4, errCache = null;
  function cfdFields(c) {
    var f = { u: Float64Array.from(c.u), v: Float64Array.from(c.v), p: Float64Array.from(c.p) };
    f.speed = f.u.map(function (a, i) { return Math.hypot(a, f.v[i]); });
    return f;
  }
  function relL2(a, b) {
    var s = 0, n = 0;
    for (var i = 0; i < a.length; i++) { s += (a[i] - b[i]) * (a[i] - b[i]); n += b[i] * b[i]; }
    return Math.sqrt(s / n);
  }
  function panel(cv, arr, r, title) {
    var dpr = window.devicePixelRatio || 1, W = Math.max(200, cv.parentElement.clientWidth);
    var l = 40, t = 28, pl = W - l - 10, H = pl + t + 66;
    cv.style.height = H + "px"; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    var g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    var N = state.N, lut = r.div ? LUT_DIV : LUT_SEQ, sp = r.hi - r.lo, ink = css("--ink");
    var oc = document.createElement("canvas"); oc.width = oc.height = N;
    var o = oc.getContext("2d"), im = o.createImageData(N, N);
    for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
      var q = Math.max(0, Math.min(255, Math.round((arr[j * N + i] - r.lo) / sp * 255))) * 3, k = ((N - 1 - j) * N + i) * 4;
      im.data[k] = lut[q]; im.data[k + 1] = lut[q + 1]; im.data[k + 2] = lut[q + 2]; im.data[k + 3] = 255;
    }
    o.putImageData(im, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(oc, l, t, pl, pl);
    g.strokeStyle = ink; g.fillStyle = ink; g.strokeRect(l - 0.5, t - 0.5, pl + 1, pl + 1);
    g.font = "600 14px system-ui, sans-serif"; g.textAlign = "left"; g.fillText(title, l, 18);
    g.font = "11.5px system-ui, sans-serif";
    [0, 0.5, 1].forEach(function (a) {
      g.textAlign = "center"; g.fillText(a.toFixed(1), l + a * pl, t + pl + 14);
      g.textAlign = "right"; g.fillText(a.toFixed(1), l - 5, t + (1 - a) * pl + 4);
    });
    g.textAlign = "center"; g.fillText("x/L", l + pl / 2, t + pl + 28);
    g.textAlign = "left"; g.fillText("y/L", 2, t + 10);
    var y0 = t + pl + 38;
    for (var s = 0; s < pl; s++) { var c = Math.round(s / (pl - 1) * 255) * 3; g.fillStyle = "rgb(" + lut[c] + "," + lut[c + 1] + "," + lut[c + 2] + ")"; g.fillRect(l + s, y0, 1.5, 10); }
    g.strokeRect(l - 0.5, y0 - 0.5, pl + 1, 11); g.fillStyle = ink; g.textAlign = "center";
    var nt = niceTicks(r.lo, r.hi, 4);
    nt.ticks.forEach(function (tv) { g.fillText(tv.toFixed(nt.dec), l + (tv - r.lo) / sp * pl, y0 + 24); });
  }
  function buildCmp() {
    var box = $("cmpChips"); box.innerHTML = "";
    CFD.cases.forEach(function (c, i) {
      var b = document.createElement("button"); b.type = "button"; b.textContent = "Re " + c.Re;
      b.setAttribute("aria-pressed", i === cmpIdx ? "true" : "false");
      b.addEventListener("click", function () { cmpIdx = i; buildCmp(); drawCompare(); });
      box.appendChild(b);
    });
    var net = state.net, rows = "<tr><th>Re</th><th>Range</th><th>Speed</th><th>u</th><th>v</th><th>p</th></tr>";
    CFD.cases.forEach(function (c, i) {
      var ai = CavityNN.newFields(state.N); CavityNN.evalGrid(net, c.Re, ai); var cf = cfdFields(c);
      rows += '<tr class="' + (i === cmpIdx ? "sel" : "") + '"><td>' + c.Re + "</td><td>" + (c.Re > net.trainMax ? "extrapolation" : "trained range") + "</td>" +
        ["speed", "u", "v", "p"].map(function (f) { return "<td>" + relL2(ai[f], cf[f]).toFixed(3) + "</td>"; }).join("") + "</tr>";
    });
    $("cmpTable").innerHTML = rows;
  }
  function drawCompare() {
    if (!state.net) return;
    var c = CFD.cases[cmpIdx], N = state.N, f = state.field, ai = CavityNN.newFields(N);
    CavityNN.evalGrid(state.net, c.Re, ai);
    var cf = cfdFields(c), both = new Float64Array(2 * N * N); both.set(cf[f]); both.set(ai[f], N * N);
    var r = fieldRange(f, both), diff = ai[f].map(function (a, i) { return a - cf[f][i]; });
    panel($("cvCfd"), cf[f], r, "CFD reference"); panel($("cvAi"), ai[f], r, "Neural-network prediction");
    panel($("cvDiff"), diff, fieldRange("u", diff), "Difference (network \u2212 CFD)");
    var mx = 0; diff.forEach(function (d) { mx = Math.max(mx, Math.abs(d)); });
    $("cmpStats").innerHTML = "<b>" + FIELD_LABEL[f] + " at Re = " + c.Re + "</b> (" +
      (c.Re > state.net.trainMax ? '<span class="warn">outside the training range 20–' + state.net.trainMax + "</span>" : "inside the training range") +
      "). Relative L2 error " + relL2(ai[f], cf[f]).toFixed(3) + ", largest absolute difference " + mx.toFixed(3) + ".";
    Array.prototype.forEach.call($("cmpTable").rows, function (row, i) { row.className = (i - 1 === cmpIdx) ? "sel" : ""; });
  }

  /* ---------- start ---------- */
  var resizeTimer = null;
  if (window.ResizeObserver) {
    new ResizeObserver(function () { if (state.data) { draw(); drawCompare(); } }).observe(wrap);
  } else {
    window.addEventListener("resize", function () { if (state.data) draw(); });
  }
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var redraw = function () { if (state.data) draw(); };
    if (mq.addEventListener) mq.addEventListener("change", redraw);
  }

  try {
    var embedded = window.CAVITY_MODEL;
    useModel(embedded, "embedded model");
  } catch (err) {
    setMessage("The embedded model could not be used: " + err.message + " Load a model file instead.", true);
  }
})();
