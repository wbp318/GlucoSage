# Insulin Management System

This project is a comprehensive insulin management system that includes glucose data analysis, glucose level prediction, and insulin dose calculation. It's designed to showcase advanced Python programming skills, data analysis, machine learning, and API development.

## Features

1. Glucose Data Analysis
   - Daily average glucose level calculation and visualization
   - Statistical analysis of glucose levels
   - Identification of hypoglycemic and hyperglycemic events
   - Time series analysis of glucose patterns
   - Clustering analysis of glucose levels

2. Glucose Level Prediction
   - Machine learning model to predict future glucose levels
   - Feature importance analysis
   - Model evaluation metrics

3. Insulin Dose Calculator API
   - Calculate insulin doses based on current glucose levels, target levels, and planned carbohydrate intake
   - Predict future glucose levels based on calculated insulin doses

4. Web Interface
   - FastAPI-based web interface to access all functionalities
   - Interactive API documentation

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/insulin-management-system.git
   cd insulin-management-system
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   python main.py
   ```

2. Open a web browser and navigate to `http://localhost:8000` to access the web interface.

3. Use the `/docs` endpoint to interact with the API directly.

## Docker Deployment

To deploy the application using Docker:

1. Build the Docker image:
   ```
   docker build -t insulin-management-system .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 insulin-management-system
   ```

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.