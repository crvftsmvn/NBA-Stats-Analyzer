# 🏀 NBA Stats Analyzer

A modern, feature-rich Flask web application for comprehensive NBA team and player statistics analysis.

## ✨ Features

- **Team Comparison**: Compare performance metrics between two teams
- **Player Analysis**: Deep dive into individual player statistics with visualizations
- **Top Performers**: Rank players based on points, rebounds, and assists
- **Game-by-Game Stats**: Track performance trends over time
- **Predictive Analytics**: Get predictions for future player performance
- **Modern UI**: Beautiful, responsive design with smooth animations

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/crvftsmvn/NBA-Stats-Analyzer.git
cd NBA-Stats-Analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser to `http://127.0.0.1:5000`

## 📊 Data Format

The application expects NBA game data in CSV format with the following structure:
- Date, Home, Away, HomeP, AwayP columns for game results
- HomeD and AwayD columns containing player statistics in list format: `[Name, Minutes, Points, Rebounds, Assists]`

## 🌐 Deployment

This app is pre-configured for deployment on [Render.com](https://render.com) using the included `render.yaml` configuration file.

### Deploy to Render

1. Connect your GitHub repository to Render
2. Render will automatically detect the `render.yaml` configuration
3. Your app will be live at `https://your-app-name.onrender.com`

## 🛠️ Technologies Used

- **Flask**: Web framework
- **Pandas**: Data processing and analysis
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning for predictions
- **Chart.js**: Interactive data visualizations
- **Tailwind CSS**: Modern, responsive styling

## 📝 License

This project is open source and available for educational purposes.

---

Made with ❤️ for NBA fans and data enthusiasts

