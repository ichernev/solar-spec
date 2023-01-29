About
=====

This repo contains a collections of tools for planning solar/off-grid
installations. The main component is a vendor specification ingestion pipeline.

Directory structure:
```
tools/               # python code
pipelines/           # descriptions of ingestion pipelines
specs/               # pydantic models for target data for ingestion
schemas/             # generated json schemas from specs/
data/                # extracted data, stored in the repo
 `- {type}/{vendor}/ # neat file hierarcy
   `- download/      # downloaded artefacts (pdf datasheets, html)
   `- extract/       # extracted tables from artefacts in CSV
   `- map/           # mapped data to vendor neutral/universal json
                     # specifications for those live under specs/
```

Rationale
=========

There are hundreds of brands of solar panels, inverters, chargers, batteries
etc. Choosing the right components for a complete installation requires
matching a bunch of attributes (i.e voltage, current of panels to inputs of
inverter).

Changing one piece might bring a cascade of other changes. For example if you
have a fixed space for panels (i.e roof) and if you change the panel model, it
might fit more or less panels, generate different voltage/currents, and you'll
have to re-match to your inverter.

The computations are not hard, but they are tiresome and should ideally be
automated. Ideally this project should provide a list of possible setups that
match given criteria (roof size, brand, grid voltage, price).

To get there the data from all vendors needs to be in a machine friendly form.

Data ingestion pipeline
-----------------------

Most vendors offer PDFs with tables that describe multiple models with similar
(but different) characteristics.
- the data (pdfs, html) is first downloaded (download stage, code is in
  tools/pipeline.py)
- then the tables are extracted and stored in CSV files (extract stage, code in
  `tools/pdf_ex.py` and `tools/html_ex.py`)
- then the data from the CSV is parsed into individual device JSON files
  (mapper stage, `tools/mapper.py`)
- data across all vendors should be interoperable, so we need a "standard"
  definition for each device class (like solar panel, solar charge controller,
  inverter, hybrid inverter, battery). The standard definitions are written in
  pydantic models (under specs/).

Usage
=====

    # prep env (only once)
    python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

    # generate json-schema (needed by mapper)
    python -m specs
    # execute all pipelines (i.e download, extract, map)
    python -m tools.pipeline
    # take a look at generated files under data/**/map/*.json

NOTE: Currently all files under data/ are commited to speed-up development and
enable (cheap) debugging/version tracking. Ideally they would live outside.

TODO
====

This is far from complete. All criticism should be formatted as a PR.

Limitations:
- image-only pdfs are not supported (OCRMyPDF might help, but that's not
  enough)
- automatic table location detection (could be added)
- the heart of the pipeline is a PDF-to-table extractor that is smart enough to
  figure out merged columns (property shared across models), row sections, etc.
  There are a ton of things that can break it, this extractor will evolve with
  time (or be scrapped and forgotten)
- add more device classes (solar charge controller, battery, pv inverter,
  inverter, charger)

License NOTE
============

The files under data are either auto-generated or downloaded from vendor (or
3rd party) and are thus not covered by the license.
