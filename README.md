# Transfer Learning for PV Forecasting under Data Scarcity: Error-Based and Economic Evaluation of NWP-Driven Hybrid Models

## Abstract

Accurate photovoltaic (PV) power forecasting is essential for the economic integration of PV systems into electricity markets. For day-ahead market participation, operators must submit reliable schedules before delivery. Data-driven and hybrid models often achieve high accuracy, but require extensive historical power and weather data from the target system. For newly installed or newly integrated PV plants, these data are initially unavailable, resulting in a cold-start problem.

This thesis investigates transfer learning for PV forecasting under data scarcity and cold-start conditions. A hybrid transfer-learning model fed with numerical weather prediction (NWP) features is developed and compared with three realistic strategies: a physical model, a generic source model trained on source plants, and a target model trained only on incoming target-system data. In contrast to previous approaches, a larger geographical distance between source and target systems is introduced to the transfer-learning setup. All models are evaluated using nMAE and nRMSE and economically through a day-ahead dispatch optimization with a battery storage system.

The generic source model provides the weakest forecasts with an nRMSE of 12.8%, due to its direct application without target adaptation. The physical model improves the nRMSE to 11.7%, but does not improve economic performance. Compared with total electricity procurement costs of 422€ without PV over the 90-day evaluation period, both models reduce total costs to 34€ without receiving a fixed feed-in tariff.

Target data collected during the cold-start phase are used to train a target model and adapt the source model through transfer learning. During the first 60 days, both data-driven approaches achieve similar accuracy, with an nRMSE of 9.9% for the target model and 9.5% for the transfer-learning model. After relearning stops, the target model fails to generalize and deteriorates over the final 30 days, resulting in an overall nRMSE of 10.3% and procurement cost of 33€, similar to the other reference models.

The transfer-learning model shows no deterioration after the learning stop and achieves the best overall accuracy and economic performance, with an nRMSE of 9.0% and procurement cost of 27€. It is therefore the most effective strategy for the investigated cold-start setting and for data-scarcity situations with up to 60 days of system-specific data. Day-ahead market participation remains beneficial compared with a feed-in tariff of 5ct/kWh, but is no longer advantageous from 6ct/kWh onward.

The economic evaluation further shows that overforecasting is more harmful than underforecasting, with an average imbalance-related cash flow of -0.14€/kWh compared with +0.06€/kWh for underforecasts. Deviations leading to unplanned imports are the most costly, while unexpected exports cause little economic damage. PV forecasting models for day-ahead dispatch should therefore penalize overforecasts more strongly.

## Code Documentation

This section provides an overview of the software provided. For detailed information on how it works and references for the methodological choices made in implementing the algorithms in this project, please refer to the thesis.

Detailed graphical documentation of the process flows can be viewed at: [Documentation Board](https://miro.com/welcomeonboard/NWZVYzl2VnZWUjEwbzR6RHVPODhha3ZnVDZUV3BuSmhrak5SVzBkSURkeUErNXZtQjJIWXlYNkc5czJmQWZoTklrZXpUV0d4b2xVSi9aQ2svV0FoZVZMQ2FLNkVGMm9WVkdRNWJHZUhyeUJaR0dXQVpsWHJjeE1tQjdpN1pGM2pQdGo1ZEV3bUdPQWRZUHQzSGl6V2NBPT0hdjE=?share_link_id=397484122180)

This documentation is structured as follows:

**Data sources** of the used Datasets are shown as flat cylinders

**Solid lines with arrows** represent automatic process flows.

**Dotted lines** indicate manual file transfers.

The project folders are shown as background boxes in the documentation:
- **data**: contains all documents and databases (csv in green, grib2 in orange, pt in
red) in different sub-folders for the processing procedure (sub-folders shown with green
background boxes). Most of the files are not uploaded in this repository, but can be recreated through the process chain.
- **services**: Created docker services that ran independently for several months in
preparation for data collection on a workstation (shown as purple box).
- **src**: Source code with all Python scripts (blue), organized in sub-folders for different stages of the process chain (shown
as gray boxes)
- **configs**: Collection of all config files (YAML and txt in pink) for parameterizing
the Python scripts (shown as pink box)
- **reports**: Graphic outputs as png (shown in yellow boxes)



# License

This project is licensed under the MIT License - see the  [LICENSE](LICENSE) file for details.
