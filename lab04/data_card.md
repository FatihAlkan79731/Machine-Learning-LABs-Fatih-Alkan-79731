# Data Card – UCI Adult (Census Income)

## Overview

| | |
|---|---|
| Name | Adult (also called "Census Income") |
| Task | Binary classification: is annual income above \$50K? |
| Source | UCI Machine Learning Repository, dataset 2. Loaded from OpenML (`fetch_openml(name='adult', version=2)`, OpenML id 1590) |
| Original data | Extracted by Barry Becker from the 1994 US Census database; donated by Barry Becker and Ronny Kohavi in 1996 |
| Citation | Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0): sharing and adaptation are allowed with attribution |

## Size and structure

- **Rows.** 48,842 rows (people), 15 columns: 14 features and 1 label.
- **Numeric features (6, integer).** age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
- **Categorical features (8).** workclass (9 values incl. `?`), education (16), marital-status (7), occupation (15 incl. `?`), relationship (6), race (5), sex (2), native-country (42 incl. `?`).
- **Label.** `income`: `<=50K` 76.07 %, `>50K` 23.93 %.
- **Missing values.** Encoded as `?`. workclass 5.73 %, occupation 5.75 %, native-country 1.75 %.
- **Duplicates.** 52 exact duplicate rows, and 5 pairs of identical feature rows with conflicting labels.
- **Hashes.** Raw data SHA256 `8f970e20…adddc`; processed data hashes in `results/run_config.yaml`.

The original extraction kept only "reasonably clean" records: age > 16, adjusted gross income > 100, final weight > 1 and hours worked > 0. This filter is why the minimum age is 17 and the minimum hours value is 1.

## Intended use

Teaching and benchmarking of tabular ML pipelines, data validation and fairness analysis. It is **not** suitable for real decisions about individuals (credit, hiring, insurance), and it is not a current picture of income in the US.

## Known biases and limitations

- **Imbalanced demographics.**
  - Sex: 66.9 % male, 33.1 % female.
  - Race: 85.5 % White, 9.6 % Black, 3.1 % Asian-Pac-Islander, 1.0 % Amer-Indian-Eskimo, 0.8 % Other.
  - Native country: 89.7 % United-States.
  - Groups this small give unreliable estimates for any model trained on the data.
- **Different label base rates between groups.**
  - By sex: 30.4 % of men vs. 10.9 % of women earn >50K.
  - By race: 25.4 % White and 26.9 % Asian-Pac-Islander, vs. 12.1 % Black, 11.7 % Amer-Indian-Eskimo and 12.3 % Other.
  - A model will learn these historical gaps and can reproduce them as unequal error rates or predictions (see `results/positive_rate_by_group.csv`).
- **Dated and not inflation-adjusted.** The data is from 1994, so the \$50K threshold is worth much more today, and occupation categories and household structures have changed since.
- **Top-coding.** capital-gain is capped at 99,999, so very high incomes are censored.
- **Missing values are not random.** `?` in workclass and occupation is concentrated in people outside regular employment, so imputing the mode (Private / Prof-specialty) adds a systematic error for that group.
- **Sampling weight.** `fnlwgt` is a census sampling weight, not a personal attribute. Using it as a plain feature has no clear meaning.
- **Label noise.** 5 pairs of identical people have opposite labels, and 4 rows have inconsistent relationship/sex values.

## Privacy

- The data is de-identified: it has no names, addresses or ID numbers. It is public under CC BY 4.0.
- It still contains **sensitive attributes**: sex, race, marital status, native country and income.
- Combinations of quasi-identifiers (age + sex + race + native-country + occupation + education) can make a person unique. This is especially true for rare groups, which raises the re-identification risk if the data were joined with other sources.
- The label (income) is financial information. Outputs of models trained on this data should not be attributed to real individuals.
- Processed files are stored only under `data/`, which git ignores; only aggregated results are committed.

## Maintenance

The source is static and has not changed since 1996. Any change to the processed data in this repository shows up as a different SHA256 in `results/run_config.yaml`.
