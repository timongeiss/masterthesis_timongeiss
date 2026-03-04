# Description

Abstract.

# Documentation

This section provides an overview of the software provided. For detailed information on how it works and references for the methodological choices made in implementing the algorithms in this project, please refer to the thesis.

Detailed graphical documentation of the process flows can be viewed at:

https://miro.com/app/board/uXjVGadA7lw=/?share_link_id=890883813480

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
