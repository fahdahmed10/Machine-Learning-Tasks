# 📚 AI-Powered Book Recommender System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

*Discover your next favorite book with the power of machine learning*

[Demo](#demo) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Contributing](#contributing)

</div>

---

## 🎯 Overview

In today's fast-paced world, finding the perfect book can be overwhelming with millions of options available. Our **AI-Powered Book Recommender System** uses **K-Nearest Neighbors (KNN) clustering algorithm** to implement collaborative filtering, finding users with similar reading preferences and recommending books they loved.

> **Why KNN for Collaborative Filtering?** It creates precise user clusters based on rating patterns and uses Euclidean distance to find your reading "neighbors" - users who rate books just like you do!

## ✨ Features

- 🤖 **KNN-Based Collaborative Filtering** - Uses K-Nearest Neighbors clustering algorithm
- 📏 **Euclidean Distance Matching** - Precise similarity calculations between users
- 🎨 **Beautiful UI** - Clean, modern Streamlit interface
- ⚡ **Real-time Clustering** - Instant user similarity detection and recommendations
- 👥 **Community-Driven** - Leverages collective user preferences and ratings
- 🎯 **Optimized K Parameter** - Fine-tuned neighbor count for best results

## 🧠 How It Works

### Collaborative Filtering with K-Nearest Neighbors

Our system uses **Collaborative Filtering** implemented through the **K-Nearest Neighbors (KNN)** algorithm - a powerful clustering approach that finds similar users based on their book ratings and preferences.

> *"Find users who rated books similarly to you, then recommend books they loved that you haven't read yet."*

| Step | Process | Description |
|------|---------|-------------|
| **1** | 📊 **Create User-Item Matrix** | Build matrix of users × books with ratings |
| **2** | 📏 **Calculate Distances** | Use Euclidean distance to measure user similarity |
| **3** | 👥 **Find K-Nearest Neighbors** | Identify the most similar users (clusters) |
| **4** | 📚 **Generate Recommendations** | Suggest highly-rated books from similar users |

### K-Nearest Neighbors Algorithm Flow

```mermaid
graph LR
    A[Selected Book] --> B[Find All Users Who Rated This Book]
    B --> C[Calculate Euclidean Distances Between Users]
    C --> D[Find K Nearest Neighbors]
    D --> E[Get Their Top-Rated Books]
    E --> F[Rank & Display Recommendations]
```

### Why KNN for Collaborative Filtering?

- **🎯 Accurate Clustering** - Groups users with similar reading preferences
- **📏 Distance-Based** - Uses Euclidean distance for precise similarity measurement  
- **⚡ Efficient** - Fast computation for real-time recommendations
- **🔄 Adaptable** - Easy to tune K parameter for optimal results

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Books-Recommender-System-Using-Machine-Learning.git
   cd Books-Recommender-System-Using-Machine-Learning
   ```

2. **Create virtual environment**
   ```bash
   # Using conda
   conda create -n book-recommender python=3.7.10 -y
   conda activate book-recommender
   
   # Or using venv
   python -m venv book-recommender
   source book-recommender/bin/activate  # On Windows: book-recommender\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate ML models**
   ```bash
   jupyter notebook "Books Recommender.ipynb"
   ```

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

## 📱 Demo

## 📱 Demo

## 📱 Demo

### 🖥️ Book Selection Interface
<div align="center">
  <img src="screenshots/book-selection.png" alt="Book Selection Interface" style="max-width: 100%; height: auto;"/>
  <p><em>Clean dropdown interface to search and select books from the database</em></p>
</div>

### 📚 AI-Powered Recommendations
<div align="center">
  <img src="screenshots/recommendations.png" alt="Book Recommendations with Covers" style="max-width: 100%; height: auto;"/>
  <p><em>Get personalized recommendations with book covers and detailed information - here showing Harry Potter series recommendations</em></p>
</div>

## 💻 Usage

1. Open your browser and navigate to `http://localhost:8501`
2. **Select a book** from the dropdown menu that you've enjoyed
3. Click **"Show Recommendation"** button
4. Get instant recommendations based on users with similar reading preferences
5. Explore book covers and details of recommended books

## 📁 Project Structure

```
📦 Books-Recommender-System/
├── 📄 app.py                    # Main Streamlit application
├── 📓 Books Recommender.ipynb   # Model training notebook
├── 📋 requirements.txt          # Python dependencies
├── 🤖 model.pkl                 # Trained ML model (generated)
├── 📊 data/                     # Dataset directory
├── 📸 screenshots/              # Application screenshots
│   ├── homepage.png
│   ├── recommendations.png
│   └── search-filter.png
├── 📚 README.md                 # This file
└── 🔧 utils/                    # Utility functions
```

## 🛠 Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) |
| **Algorithm** | ![KNN](https://img.shields.io/badge/K--Nearest_Neighbors-4CAF50?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiLz4KPC9zdmc+&logoColor=white) ![Euclidean Distance](https://img.shields.io/badge/Euclidean_Distance-2196F3?style=flat) |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| **Development** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) |

</div>

## 📊 Dataset

The recommendation system is built using the comprehensive **Book Recommendation Dataset** from Kaggle:

- **Source**: [Book Recommendation Dataset](https://www.kaggle.com/ra4u12/bookrecommendation)
- **Size**: 1M+ book ratings
- **Features**: Book metadata, user ratings, reviews

## 🔮 Future Roadmap

- [ ] **Optimize K Parameter** - Implement cross-validation to find optimal K value
- [ ] **Advanced Distance Metrics** - Experiment with cosine similarity and Manhattan distance
- [ ] **Weighted KNN** - Give more weight to closer neighbors in recommendations
- [ ] **User Profiles** - Add authentication and rating history
- [ ] **Real-time Model Updates** - Retrain KNN model as new ratings come in
- [ ] **Performance Optimization** - Use approximate nearest neighbors for large datasets
- [ ] **API Integration** - Connect with book databases for more user rating data

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💻 Make your changes
4. ✅ Run tests and ensure code quality
5. 📝 Commit your changes (`git commit -m 'Add amazing feature'`)
6. 🚀 Push to the branch (`git push origin feature/amazing-feature`)
7. 🎯 Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the Kaggle community for providing the dataset
- Inspired by modern recommendation systems used by Netflix, Amazon, and Spotify
- Built with ❤️ by the open-source community

---

<div align="center">

**Built with ❤️ by [Fahd Ahmed Ali](https://www.linkedin.com/in/fahd-ahmed-9b6755307/)**

*Data & ML Engineer | Turning data into intelligent recommendations*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fahd-ahmed-9b6755307/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:afahd9002@gmail.com)

</div>

---

<div align="center">
  <sub>⭐ If you found this project helpful, please give it a star!</sub>
</div>