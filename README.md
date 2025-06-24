# Kernel Regression Comparison

This project implements and compares two kernelized regression methods— **Kernel Ridge Regression (KRR)** and **ε-Support Vector Regression (SVR)**— using the Polynomial kernel, RBF kernel, and a custom 2-gram spectrum kernel for structured string data.

## 📁 Project Layout

```
├── data/
│   ├── sine.csv                # Part 1 data
│   ├── housing2r.csv          # Part 2 data
│   ├── part3_data.csv         # Part 3 full dataset
│   ├── part3_train.csv        # Part 3 train split
│   └── part3_test.csv         # Part 3 test split
├── plots_part1/               # SVG plots from Part 1
├── plots_part2/               # SVG plots from Part 2
├── plots_part3/               # SVG plots from Part 3
├── report/
│   └── report.pdf             # Your write-up
├── src/
│   ├── hw_kernels.py          # All implementations & pipelines
│   └── test_kernels.py        # Unit tests (pytest)
├── .gitignore
└── README.md
```

## 🚀 Getting Started

1. **Install prerequisites**

```bash
pip install numpy pandas matplotlib scikit-learn cvxopt pytest
```

2. **Run all three parts & generate plots**

```bash
cd src
python hw_kernels.py
```

The scripts will save SVGs to `../plots_part1/`, `../plots_part2/`, and `../plots_part3/`.

3. **Run the unit tests**

```bash
cd src
pytest test_kernels.py
```

## 🛠️ Usage Details

* **Part 1**: Fits sine data with KRR & SVR (Polynomial & RBF kernels), highlights support vectors.
* **Part 2**: Sweeps kernel hyperparameters on `housing2r.csv`, compares fixed vs CV-tuned regularization, and plots MSE & SV counts.
* **Part 3**: Builds a 2-gram string kernel vs naive character counts, compares test MSE.

## 📄 Report

See `report/report.pdf` for methodology, quantitative results, and discussion.

## 📜 License

Released under the MIT License.