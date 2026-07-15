# Garak evaluation

This directory contains the configuration for evaluating Palmo's local Flask API with [Garak](https://github.com/NVIDIA/garak).

Start the API in a separate terminal:

```bash
python app.py
```

Then run Garak with the included REST generator configuration:

```bash
garak --config garak/garak_palmo.json
```

The configuration expects the endpoint at `http://127.0.0.1:8001/generate` and reads generated text from the `text` JSON field.
