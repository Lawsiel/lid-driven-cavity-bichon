# Lid-driven cavity: neural-network surrogate with CFD extrapolation test

Static site, no build step and no server code. Student: Chiara (Week 9).

- `index.html`: page structure
- `style.css`: styling (light and dark)
- `js/cavity-core.js`: network inference and verification against the Python outputs
- `js/app.js`: heatmap, streamlines, controls, CFD comparison
- `data/model-data.js`: exported network weights (Re 20-300 training range)
- `data/cfd-data.js`: CFD reference cases (Re 100-1000), for checking only

Open `index.html` locally, or publish on GitHub Pages (Settings > Pages > Deploy from a branch > main, / (root)).
