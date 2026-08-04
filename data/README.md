# Data Instructions

## Dataset

Use the UCI ElectricityLoadDiagrams20112014 dataset.

Dataset page and citation:

https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

Trindade, A. (2015). ElectricityLoadDiagrams20112014 [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C58C86

## Where To Put The Raw File

Download and place the raw data file under:

```text
data/raw/
```

Expected common filename:

```text
LD2011_2014.txt
```

If your downloaded file has a different name, either rename it to `LD2011_2014.txt` or update `RAW_DATA_FILENAME` in `src/config.py`.

## Notes

- The data is recorded at 15-minute intervals.
- Timestamps use Portuguese local time, so the default holiday calendar is
  Portugal (`PT`). Switch it to Hong Kong (`HK`) only when using Hong Kong data.
- Values are electricity demand in kW. Divide by 4 to convert each 15-minute
  reading to kWh.
- The expected SHA256 for the downloaded text file used in this project is
  `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`.
- Forecasting 1 hour ahead means forecasting 4 steps ahead.
- Forecasting 24 hours ahead means forecasting 96 steps ahead.
- Raw data files are not committed to GitHub because they can be large.
