# PV Forecasting under Data Scarcity - Research Code and Reproduction

Software accompanying the master's thesis **"Transfer Learning for PV Forecasting under Data Scarcity: Error-Based and Economic Evaluation of NWP-Driven Hybrid Models"** and the follow-up manuscript **"Transfer Learning for Photovoltaic Forecasting under Data Scarcity: Technical and Economic Value of Generic Multi-System Pretraining under Domain Shift."**

This repository contains the research code and configuration files used to develop, train, adapt, and evaluate photovoltaic (PV) forecasting models in a cold-start setting and to assess their use in day-ahead PV–battery dispatch. It is intended as a resource for understanding and reproducing the computational workflow, rather than as a repository of published results.

## From the Master's Thesis to the Follow-up Paper

The two works originate from the same evolving research project, but address it at different levels of focus.

**Master's thesis: development and broader assessment.**

The thesis documents the construction of an end-to-end, numerical-weather-prediction (NWP)-driven PV forecasting and evaluation pipeline. It brings together physical forecasting, a generic multi-system source model, target-only learning, transfer learning, technical error analysis, and a market-oriented PV–battery dispatch model. Its broader scope reflects the development of the software, data infrastructure, and experimental framework.

**Follow-up manuscript: a focused applied research question.**

Building on that framework, the paper concentrates on a decision faced by operators integrating a new PV asset: can a model pretrained on a generic portfolio be adapted successfully when little target data are available, even if the source and target differ geographically, in PV-system characteristics, and in weather-data products?

The paper organizes the comparison around realistic cold-start options, distinguishes daily adaptation from subsequent operation without parameter updates, and relates forecast quality to economic value in day-ahead dispatch.

The shift in storyline is therefore one of research focus and presentation, not two unrelated projects: the thesis describes the broader development and investigation, while the manuscript distills the same underlying work into a more specific, application-oriented contribution. Both are supported by the software in this repository.

## Implemented Workflow

The repository supports the main stages of the research pipeline:

1. **Data acquisition and preparation:** Process measured PV generation, plant metadata, meteorological reanalysis for source-domain training, and recorded operational NWP inputs for the target domain.
2. **Source-model pretraining:** Train a generic multilayer perceptron (MLP) on measurements from multiple PV systems, using weather and plant-related features.
3. **Cold-start forecasting strategies:** Generate forecasts using a pvlib-based physical baseline, an unadapted source model, a target-only MLP trained from local observations, and a transfer-learning MLP adapted from the pretrained source representation.
4. **Sequential evaluation:** Reproduce the progressive target-data setting, daily model adaptation, and a separate no-update phase to assess retained generalization.
5. **Economic evaluation:** Use PV forecasts for day-ahead PV–battery scheduling and evaluate the resulting deviations using realized PV generation and historical imbalance prices.

The follow-up manuscript specifically examines generic pretraining on a portfolio of 172 real PV systems with reanalysis weather inputs and transfer to a geographically distant target forecast with operational NWP inputs. The physical and unadapted source strategies do not require target-specific model fitting; the target-only and transfer-learning strategies are updated as local training data become available.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/` | Python scripts for data processing, modeling, forecasting, and evaluation. |
| `configs/` | YAML and text configuration files used to parameterize scripts. |
| `services/` | Docker-based services used for longer-running data collection. |
| `data/` | Input, intermediate, and processed data locations; large datasets are not fully included. |
| `reports/` | Location for generated figures and other reporting outputs. |
| `CITATION.cff` | Software citation metadata. |
| `LICENSE` | MIT license. |

## Reproduction and Data Availability

Start with the scripts in `src/`, their corresponding settings in `configs/`, and the [workflow documentation board](https://miro.com/welcomeonboard/NWZVYzl2VnZWUjEwbzR6RHVPODhha3ZnVDZUV3BuSmhrak5SVzBkSURkeUErNXZtQjJIWXlYNkc5czJmQWZoTklrZXpUV0d4b2xVSi9aQ2svV0FoZVZMQ2FLNkVGMm9WVkdRNWJHZUhyeUJaR0dXQVpsWHJjeE1tQjdpN1pGM2pQdGo1ZEV3bUdPQWRZUHQzSGl6V2NBPT0hdjE=?share_link_id=397484122180), which illustrates the process flow and identifies automated and manual data-transfer steps.

This is a research codebase, not a self-contained one-command reproduction package. The software archive does not contain all inputs needed for an exact rerun.
In particular, large meteorological archives and intermediate datasets are not bundled in full. Reproduction requires the appropriate PV measurements, meteorological inputs, market-price series, and experiment configurations.

Source-domain PV data and reanalysis products are described in the accompanying research documents. Operational target NWP forecasts were collected separately. Availability and access conditions for processed study data should be checked in the corresponding publication or data release.

The purpose of this repository and its software archive is to provide the implementation and reproduction workflow, not to duplicate the manuscript's figures, tables, or numerical findings.

## Related Research and Citation

- **Master's thesis:** Geiss, T. (2026). Transfer Learning for PV Forecasting under Data Scarcity: Error-Based and Economic Evaluation of NWP-Driven Hybrid Models. Master's thesis.
- **Follow-up manuscript:** Details will be added when available.

For use of the code, cite the archived software version using the metadata in `CITATION.cff` or the DOI assigned to the relevant Zenodo release. For scientific methods and interpretations, cite the corresponding thesis or manuscript.

## License

This project is distributed under the [MIT License](LICENSE).
