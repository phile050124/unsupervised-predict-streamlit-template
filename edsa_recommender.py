import streamlit as st
import pandas as pd


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

# Set the background color for the menu list to "mad red"
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #ff0000; /* Mad red color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the background color for the app to "dark blue"
st.markdown(
    """
    <style>
        .reportview-container {
            background-color: #00008B; /* Dark blue color */
            color: white; /* Text color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app declaration
def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Search Movies", "Top Charts", "User Profile"]

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
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend", key="content_based"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm doesn't work. \
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend", key="collaborative_based"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm doesn't work. \
                              We'll need to fix it!")

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    elif page_selection == "Search Movies":
        st.title("Search Movies")
        search_query = st.text_input("Enter the title of the movie you want to search:")
        if search_query:
            search_results = title_list[title_list['title'].str.lower().str.contains(search_query.lower())]
            if not search_results.empty:
                st.write("Search Results:")
                st.table(search_results[['title', 'genres', 'release_date', 'actors', 'duration']])
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

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
