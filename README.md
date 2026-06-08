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

Clone the repository:

```bash
git clone https://github.com/silviodavidrodriguez/web-of-statistics.git
cd web-of-statistics
```

Create a virtual environment:

```bash
python -m venv proyectoenv
```

Activate the environment:

### Windows

```bash
proyectoenv\Scripts\activate
```

### Linux

```bash
source proyectoenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run database migrations:

```bash
python manage.py migrate
```

Start the development server:

```bash
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000
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
