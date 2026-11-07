/* ======================================================================
   Core: model loading, inference and verification (no DOM access)
   ====================================================================== */
var CavityNN = (function () {
  "use strict";

  // Activations by exported layer type. Anything not listed is rejected
  // loudly instead of being silently replaced by something else.
  var ACTIVATIONS = {
    tanh: function () { return Math.tanh; },
    relu: function () { return function (z) { return z > 0 ? z : 0; }; },
    leakyrelu: function (L) {
      var s = (L.negative_slope != null) ? L.negative_slope : (L.alpha != null ? L.alpha : 0.01);
      return function (z) { return z > 0 ? z : s * z; };
    },
    sigmoid: function () { return function (z) { return 1 / (1 + Math.exp(-z)); }; },
    elu: function (L) {
      var a = (L.alpha != null) ? L.alpha : 1;
      return function (z) { return z > 0 ? z : a * Math.expm1(z); };
    },
    silu: function () { return function (z) { return z / (1 + Math.exp(-z)); }; },
    softplus: function () { return function (z) { return z > 30 ? z : Math.log1p(Math.exp(z)); }; },
    identity: function () { return function (z) { return z; }; }
  };

  function sameList(a, b) {
    return Array.isArray(a) && a.length === b.length && a.every(function (v, i) { return v === b[i]; });
  }

  function compile(model) {
    if (!model || !Array.isArray(model.layers)) throw new Error("The file has no 'layers' list.");
    var inNames = model.input_order, outNames = model.output_order;
    if (!Array.isArray(inNames) || inNames.length !== 3) throw new Error("input_order must list 3 inputs.");
    ["x", "y", "Re"].forEach(function (n) {
      if (inNames.indexOf(n) < 0) throw new Error("input_order is missing '" + n + "'.");
    });
    ["u", "v", "p"].forEach(function (n) {
      if (!Array.isArray(outNames) || outNames.indexOf(n) < 0) throw new Error("output_order is missing '" + n + "'.");
    });
    ["input_mean", "input_std", "output_mean", "output_std"].forEach(function (k) {
      if (!Array.isArray(model[k]) || model[k].length !== 3) throw new Error("'" + k + "' must have 3 numbers.");
    });

    var width = 3, ops = [];
    model.layers.forEach(function (L, idx) {
      var type = String(L.type);
      if (type === "Linear") {
        var W = L.weight, b = L.bias;
        if (!Array.isArray(W) || !Array.isArray(b)) throw new Error("Layer " + idx + ": Linear needs weight and bias.");
        var nOut = W.length, nIn = W[0].length;
        if (nIn !== width) throw new Error("Layer " + idx + ": expects " + width + " inputs but weight rows have " + nIn + ".");
        if (b.length !== nOut) throw new Error("Layer " + idx + ": bias length does not match weight rows.");
        // weight[j][i] maps input i to output j  ->  flat index j * nIn + i
        var w = new Float64Array(nOut * nIn);
        for (var j = 0; j < nOut; j++) {
          if (W[j].length !== nIn) throw new Error("Layer " + idx + ": ragged weight matrix.");
          for (var i = 0; i < nIn; i++) w[j * nIn + i] = W[j][i];
        }
        ops.push({ kind: "linear", nIn: nIn, nOut: nOut, w: w, b: Float64Array.from(b), buf: new Float64Array(nOut) });
        width = nOut;
      } else {
        var make = ACTIVATIONS[type.toLowerCase()];
        if (!make) throw new Error("Layer " + idx + ": activation '" + type + "' is not supported by this demo.");
        ops.push({ kind: "act", fn: make(L), n: width });
      }
    });
    if (width !== 3) throw new Error("The last layer must output 3 values (u, v, p); it outputs " + width + ".");

    return {
      ops: ops,
      inNames: inNames.slice(),
      inMean: Float64Array.from(model.input_mean),
      inStd: Float64Array.from(model.input_std),
      outMean: Float64Array.from(model.output_mean),
      outStd: Float64Array.from(model.output_std),
      iu: outNames.indexOf("u"), iv: outNames.indexOf("v"), ip: outNames.indexOf("p"),
      x0: new Float64Array(3),
      reMin: model.reynolds_range ? model.reynolds_range[0] : 20,
      reMax: model.reynolds_range ? model.reynolds_range[1] : 300,
      N: (model.grid && model.grid.n) ? model.grid.n : 50
    };
  }

  // One forward pass. `out` receives [u, v, p] in physical units.
  function predict(net, x, y, Re, out) {
    var a = net.x0, k, i, j;
    for (k = 0; k < 3; k++) {
      var name = net.inNames[k];
      var raw = (name === "x") ? x : (name === "y") ? y : Re;
      a[k] = (raw - net.inMean[k]) / net.inStd[k];
    }
    var cur = a;
    for (k = 0; k < net.ops.length; k++) {
      var op = net.ops[k];
      if (op.kind === "linear") {
        var nIn = op.nIn, w = op.w, b = op.b, dst = op.buf;
        for (j = 0; j < op.nOut; j++) {
          var s = b[j], base = j * nIn;
          for (i = 0; i < nIn; i++) s += w[base + i] * cur[i];
          dst[j] = s;
        }
        cur = dst;
      } else {
        for (i = 0; i < op.n; i++) cur[i] = op.fn(cur[i]);
      }
    }
    // Restore physical units. No extra activation is applied to the last Linear output.
    var o = new Array(3);
    for (k = 0; k < 3; k++) o[k] = cur[k] * net.outStd[k] + net.outMean[k];
    out[0] = o[net.iu]; out[1] = o[net.iv]; out[2] = o[net.ip];
    return out;
  }

  // Evaluate at all N x N cell centres. Index = j * N + i, j counts upward in y.
  function evalGrid(net, Re, out) {
    var N = net.N, tmp = [0, 0, 0];
    for (var j = 0; j < N; j++) {
      var y = (j + 0.5) / N;
      for (var i = 0; i < N; i++) {
        var x = (i + 0.5) / N;
        predict(net, x, y, Re, tmp);
        var idx = j * N + i;
        out.u[idx] = tmp[0]; out.v[idx] = tmp[1]; out.p[idx] = tmp[2];
        out.speed[idx] = Math.sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1]);
      }
    }
    return out;
  }

  function newFields(N) {
    return { u: new Float64Array(N * N), v: new Float64Array(N * N), p: new Float64Array(N * N), speed: new Float64Array(N * N) };
  }

  // Compare the JavaScript network with the Python outputs stored in the file.
  function verify(net, model) {
    var V = model.verification;
    if (!V || !Array.isArray(V.inputs) || !Array.isArray(V.outputs) || V.inputs.length !== V.outputs.length) {
      return { available: false, pass: false, maxErr: NaN, tol: NaN, rows: [] };
    }
    var tol = (V.absolute_tolerance != null) ? V.absolute_tolerance : 1e-5;
    var outNames = model.output_order, tmp = [0, 0, 0], maxErr = 0, rows = [];
    V.inputs.forEach(function (inp, r) {
      predict(net, inp[0], inp[1], inp[2], tmp);
      var got = { u: tmp[0], v: tmp[1], p: tmp[2] };
      var err = 0;
      ["u", "v", "p"].forEach(function (n) {
        var exp = V.outputs[r][outNames.indexOf(n)];
        err = Math.max(err, Math.abs(got[n] - exp));
      });
      maxErr = Math.max(maxErr, err);
      rows.push({ x: inp[0], y: inp[1], Re: inp[2], err: err, ok: err <= tol });
    });
    return { available: true, pass: maxErr <= tol && rows.length > 0, maxErr: maxErr, tol: tol, rows: rows };
  }

  return { compile: compile, predict: predict, evalGrid: evalGrid, newFields: newFields, verify: verify };
})();
