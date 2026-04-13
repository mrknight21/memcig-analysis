# Website Component

This folder contains the annotation web interface:

- `application.py`: Flask app implementation
- `templates/`: Jinja templates
- `static/`: static assets and local website metadata files
- `annotation_tracking.py`: website-side tracking utility script

Run the app from repo root with:

```bash
flask --app application run
```

The root-level `application.py` is a compatibility shim that imports `website.application`.

