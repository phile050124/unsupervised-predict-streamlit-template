"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# Set theme colors
primary_color = '#3498db'  # Blue color
secondary_color = '#ffffff'  # White color
bg_color = '#f0f6fc'  # Light blue color for background
# Set page background color
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #00008B; /* Dark blue color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the background color for the app to "mad red"
st.markdown(
    """
    <style>
        .reportview-container {
            background-color: #ff0000; /* Mad red color */
            color: white; /* Text color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App declaration
def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Search Movies", "Top Charts", "User Profile", "About App", "About Owners"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    elif page_selection == "Search Movies":
        st.title("Search Movies")
        search_query = st.text_input("Enter the title of the movie you want to search:")
        if search_query:
            search_results = [title for title in title_list if search_query.lower() in title.lower()]
            if search_results:
                st.write("Search Results:")
                for result in search_results:
                    st.write(result)
            else:
                st.write("No matching movies found.")

    elif page_selection == "Top Charts":
        st.title("Top Charts")
        # Load movie ratings data
        ratings_df = pd.read_csv('resources/data/ratings.csv')
        # Calculate average rating for each movie
        avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
        # Sort movies based on average rating
        top_movies = avg_ratings.sort_values(ascending=False).head(10)
        st.write("Top 10 Rated Movies:")
        for movie_id, rating in top_movies.items():
            movie_title = title_list[movie_id]
            st.write(f"{movie_title}: {rating:.2f}")

    elif page_selection == "User Profile":
        st.title("User Profile")
        st.sidebar.title("User Profile Management")

        # Sidebar menu for profile management
        profile_options = ["View Profile", "Edit Profile", "Preferences"]
        selected_option = st.sidebar.radio("Select Option", profile_options)

        if selected_option == "View Profile":
            st.write("View Profile:")
            # Placeholder for viewing user profile
            st.write("Name: John Doe")
            st.write("Email: john@example.com")
            st.write("Age: 30")
            st.write("Gender: Male")

        elif selected_option == "Edit Profile":
            st.write("Edit Profile:")
            # Placeholder for editing user profile
            name = st.text_input("Name", "John Doe")
            email = st.text_input("Email", "john@example.com")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.radio("Gender", ["Male", "Female"])
            if st.button("Update Profile"):
                # Placeholder for updating user profile in database
                st.success("Profile updated successfully!")

        elif selected_option == "Preferences":
            st.write("Preferences:")
            # Placeholder for managing user preferences
            st.write("Preferred Genre: Action")
            st.write("Favorite Actor: Tom Hanks")
            st.write("Favorite Director: Christopher Nolan")

            # Rating and feedback system
            st.subheader("Rate and Provide Feedback for Movies:")
            selected_movie = st.selectbox("Select a Movie:", title_list)
            rating = st.slider("Rating (1-5)", 1, 5)
            feedback = st.text_area("Feedback")

            if st.button("Submit"):
                # Placeholder for saving rating and feedback to database
                st.success(
                    f"Thank you for rating '{selected_movie}' with {rating} stars and providing feedback:\n{feedback}")

    elif page_selection == "About App":
        st.title("About the App")
        st.write("""
            This app is designed to provide personalized movie recommendations based on user preferences and movie ratings.
            Users can log in, rate movies, and receive tailored recommendations to enhance their movie-watching experience.
            """)

    elif page_selection == "About Owners":
        st.title("About the Owners")
        st.write("""
            This app is developed by Explore Data Science Academy as part of the Unsupervised Learning Predict project.
            For more information about the owners and their projects, visit [Explore Data Science Academy](https://explore-datascience.net/).
            """)


if __name__ == '__main__':
    main()
