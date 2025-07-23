# Smart Grid Management System

## 📌 Overview
The **Smart Grid Management System** is a comprehensive solution designed to monitor, predict, and manage energy distribution efficiently.  
This project leverages **Python (Django)** for backend operations, **React.js** for frontend visualization, and incorporates **Machine Learning** for load prediction and anomaly detection.

## ✨ Features
- **Real-Time Energy Monitoring**: Visualize live energy usage and grid performance.
- **Load Prediction**: Uses machine learning models to forecast future energy demands.
- **Energy Theft Detection**: Detect unusual energy consumption patterns.
- **Load Balancing**: Optimize power distribution to prevent outages.
- **Data Visualization**: Interactive charts and dashboards for quick decision-making.
- **Secure and Scalable**: Implements authentication and scalable infrastructure.
  
## 🛠️ Tech Stack
### Backend:
- **Django (Python 3.x)**
- **Django REST Framework**
- **SQLite / PostgreSQL (configurable)**

### Frontend:
- **React.js**
- **Chart.js / Recharts** for data visualization

### Machine Learning:
- **Scikit-learn**
- **Pandas**
- **NumPy**

### Others:
- **Docker (Optional)**
- **Git & GitHub** for version control

## 📂 Project Structure
smart-grid-management/
│
├── backend/
│ ├── manage.py
│ ├── smartgrid_app/
│ ├── requirements.txt
│
├── frontend/
│ ├── package.json
│ ├── src/
│
├── README.md
└── .gitignore

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/smart-grid-management.git
cd smart-grid-management

2. Backend Setup
cd backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
3. Frontend Setup
cd ../frontend
npm install
npm start
4. Access the App
Open http://localhost:3000 to view the React frontend and
http://localhost:8000 for Django backend APIs.
5. Access the App
Open http://localhost:3000 to view the React frontend and
http://localhost:8000 for Django backend APIs.

📊 Machine Learning Model
The ML model for load prediction is trained on historical energy consumption data.

Implements regression techniques for accurate forecasting.

Scripts for training are provided in backend/ml/ directory.

🤝 Contributing
Contributions are welcome!

Fork the repository

Create a feature branch

Submit a pull request


📧 Contact
For any inquiries or collaboration:
Author: Mohammad Ayaan Sohail
Email: mohammadayaansohail@gmail.com
