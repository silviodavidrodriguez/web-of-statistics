# Web of Statistics

Free online statistical tools for students, researchers, teachers, and professionals.

🌐 Official Website: https://www.webofstatistics.com

📦 GitHub Repository: https://github.com/silviodavidrodriguez/web-of-statistics

Web of Statistics is an open-source web platform that provides interactive statistical analysis tools directly from a web browser. The project aims to make statistical methods accessible without requiring specialized software installation.

The platform includes tools for descriptive statistics, probability distributions, statistical inference, control charts, ANOVA, regression analysis, and multivariate methods.

---

## Features

### Descriptive Statistics

* Summary statistics
* Histograms
* Boxplots
* Exploratory data analysis

### Probability

* Normal distribution
* Student's t distribution
* Chi-squared distribution
* F distribution
* Statistical tables

### Statistical Inference

* Confidence intervals
* Hypothesis testing
* Goodness-of-fit tests
* Non-parametric methods

### Statistical Process Control

* Shewhart control charts
* EWMA charts
* CUSUM charts
* Precontrol charts

### ANOVA

* One-way ANOVA
* Two-way ANOVA
* Randomized block designs
* MANOVA
* Non-parametric ANOVA

### Regression

* Simple linear regression
* Multiple linear regression
* Interaction models
* Nonlinear regression

### Multivariate Analysis

* Principal Component Analysis (PCA)
* Hierarchical Clustering Analysis (HCA)
* K-Means Clustering
* Correlation matrices
* SIMCA

### Multivariate Regression

* Principal Component Regression (PCR)
* Partial Least Squares Regression (PLSR)

### Classification

* Linear Discriminant Analysis (LDA)
* Quadratic Discriminant Analysis (QDA)
* Partial Least Squares Discriminant Analysis (PLS-DA)
* Support Vector Machine (SVM)
* Logistic Regression
* Naive Bayes
* k-Nearest Neighbors Classification (k-NN)

### Tree Models

* Decision Tree Classification
* Random Forest Classification
* XGBoost Classification

---

## Technology Stack

* Python
* Django
* NumPy
* Pandas
* SciPy
* Statsmodels
* Scikit-learn
* Plotly
* PostgreSQL
* Gunicorn
* Nginx

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/silviodavidrodriguez/web-of-statistics.git
cd web-of-statistics
```

### 2. Install GNU gettext

This project uses Django's internationalization system. GNU gettext must be installed to compile the translation files located in the `locale` directory.

You can verify whether gettext is already installed by running:

```bash
gettext --version
```

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install gettext
```

#### Fedora / Rocky Linux / AlmaLinux

```bash
sudo dnf install gettext
```

#### Arch Linux

```bash
sudo pacman -S gettext
```

#### macOS

Using Homebrew:

```bash
brew install gettext
brew link --force gettext
```

#### Windows

Using Chocolatey:

```powershell
choco install gettext
```

After installation, close and reopen the terminal and verify that gettext is available:

```powershell
gettext --version
```

Make sure the gettext installation directory is included in the Windows `PATH` environment variable.

### 3. Create a virtual environment

```bash
python -m venv proyectoenv
```

### 4. Activate the virtual environment

#### Windows

Command Prompt:

```cmd
proyectoenv\Scripts\activate
```

PowerShell:

```powershell
.\proyectoenv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source proyectoenv/bin/activate
```

### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 6. Compile translation files

Compile the `.po` translation files into `.mo` files:

```bash
python manage.py compilemessages
```

This command requires GNU gettext to be installed and available in the system `PATH`.

The compiled translations will be generated inside the corresponding language directories:

```text
locale/
├── en/
│   └── LC_MESSAGES/
│       ├── django.po
│       └── django.mo
└── es/
    └── LC_MESSAGES/
        ├── django.po
        └── django.mo
```

Whenever a `.po` file is modified, run the following command again:

```bash
python manage.py compilemessages
```

### 7. Run database migrations

```bash
python manage.py migrate
```

### 8. Start the development server

```bash
python manage.py runserver
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

### Troubleshooting translations

If Django displays an error indicating that `msgfmt` cannot be found, verify that GNU gettext is installed:

```bash
msgfmt --version
```

If the command is not recognized, add the gettext `bin` directory to the system `PATH`, restart the terminal, and run:

```bash
python manage.py compilemessages
```

---

## Project Structure

```text
proyecto_stat/
│
├── home/
├── descriptive/
├── probability/
├── inference/
├── control/
├── anova/
├── regression/
├── multivariate/
│
├── static/
├── manage.py
└── proyecto_stat/
```

---

## Contributing

Contributions are welcome.

If you find bugs, have suggestions for improvements, or want to contribute new statistical tools, please open an issue or submit a pull request.

---

## Author

### Web of Statistics Project

Created and maintained by **Dr. Silvio D. Rodríguez**

Professor of Applied Statistics
Pontificia Universidad Católica Argentina (UCA)

Research Associate
CONICET (Argentina)

Contact:

* [silviodavidrodriguez@gmail.com](mailto:silviodavidrodriguez@gmail.com)
* [silviodavidrodriguez@uca.edu.ar](mailto:silviodavidrodriguez@uca.edu.ar)

LinkedIn:

* https://www.linkedin.com/in/silvio-david-rodriguez-0a573a39

---

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).

Any distributed modified version of this software must also remain open source and be distributed under the same license.
