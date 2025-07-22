# Project: Book Recommender System Using Machine Learning! | Collaborative Filtering Based

Recommendation systems are becoming increasingly important in today's extremely busy world. People are always short on time with the myriad tasks they need to accomplish in the limited 24 hours. Therefore, the recommendation systems are important as they help them make the right choices, without having to expend their cognitive resources.

The purpose of a recommendation system basically is to search for content that would be interesting to an individual. Moreover, it involves a number of factors to create personalised lists of useful and interesting content specific to each user/individual. Recommendation systems are Artificial Intelligence based algorithms that skim through all possible options and create a customized list of items that are interesting and relevant to an individual. These results are based on their profile, search/browsing history, what other people with similar traits/demographics are reading, and how likely are you to enjoy those books. This is achieved through predictive modeling and heuristics with the data available.

## Types of Recommendation System:

### 1) Content Based:

- Content-based systems, which use characteristic information and takes item attributes into consideration.

- Twitter, Youtube.

- Which books you are reading, what genres you prefer. Form embeddings for the features.
	
- User specific actions or similar items recommendation.
	
- It will create a vector of it.
	
- These systems make recommendations using a user's item and profile features. They hypothesize that if a user was interested in an item in the past, they will once again be interested in it in the future.
	
- One issue that arises is making obvious recommendations because of excessive specialization (user A is only interested in categories B, C, and D, and the system is not able to recommend items outside those categories, even though they could be interesting to them).

### 2) Collaborative Based:
		
- Collaborative filtering systems, which are based on user-item interactions.
	
- Clusters of users with same ratings, similar users.
	
- Book recommendation, so use cluster mechanism.
	
- We take only one parameter, ratings or comments.
	
- In short, collaborative filtering systems are based on the assumption that if a user likes item A and another user likes the same item A as well as another item, item B, the first user could also be interested in the second item.
	
- Issues are:

	- User-Item nXn matrix, so computationally expensive.

	- Only famous items will get recommended.

	- New items might not get recommended at all.   

### 3) Hybrid Based:
	
- Hybrid systems, which combine both types of information with the aim of avoiding problems that are generated when working with just one kind.

- Combination of both and used now a days.

- Uses: word2vec, embedding.           

## About this project:

This is a streamlit web application that can recommend various kinds of similar books based on user interest. The system utilizes machine learning algorithms to analyze book features and user preferences to provide personalized book recommendations.

## Features:

- **Intelligent Book Recommendations**: Get personalized book suggestions based on your reading preferences
- **User-Friendly Interface**: Simple and intuitive Streamlit web interface
- **Machine Learning Powered**: Uses advanced ML algorithms for accurate recommendations
- **Collaborative Filtering**: Leverages user behavior and ratings for better suggestions

## Dataset:

* [Dataset link](https://www.kaggle.com/ra4u12/bookrecommendation)

## Concept used to build the model.pkl file: NearestNeighbors

1. Load the data
	
2. Initialise the value of k

3. For getting the predicted class, iterate from 1 to total number of training data points

4. Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric since it's the most popular method.

5. Sort the calculated distances in ascending order based on distance values
	
6. Get top k rows from the sorted array

## How to run?

### STEPS:

Clone the repository

```bash
git clone https://github.com/your-username/Books-Recommender-System-Using-Machine-Learning
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n books python=3.7.10 -y
```

```bash
conda activate books
```

### STEP 02- Install the requirements

```bash
pip install -r requirements.txt
```

### STEP 03- Generate the models

```bash
# Run this file to generate the models
jupyter notebook "Books Recommender.ipynb"
```

### STEP 04- Run the application

```bash
streamlit run app.py
```

## Technologies Used:

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **NearestNeighbors**: Algorithm for recommendation system

## Project Structure:

```
├── app.py                          # Main Streamlit application
├── Books Recommender.ipynb         # Jupyter notebook for model training
├── requirements.txt                # Python dependencies
├── model.pkl                       # Trained machine learning model
└── README.md                       # Project documentation
```

## Future Enhancements:

- Add more sophisticated filtering options
- Implement hybrid recommendation approach
- Include book reviews and ratings analysis
- Add user authentication and personalization
- Deploy to cloud platform for wider accessibility

---

```bash
Author: Fahd Ahmed Ali
Data  ML Engineer
Email: afahd9002@gmail.com
LinkedIn: https://www.linkedin.com/in/fahd-ahmed-9b6755307/
```