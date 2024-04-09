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

# Import seaborn library
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# To create interactive plots
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.express as px


# Data handling dependencies
import pandas as pd
import numpy as np
import codecs

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
st.set_option('deprecation.showPyplotGlobalUse', False)
# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# Importing data
movies = pd.read_csv('resources/data/movies.csv')
train = pd.read_csv('resources/data/ratings.csv')
df_imdb = pd.read_csv('resources/data/imdb_data.csv')

# Merging the train and the movies
df_merge1 = train.merge(movies, on = 'movieId')

from datetime import datetime
# Convert timestamp to year column representing the year the rating was made on merged dataframe
df_merge1['rating_year'] = df_merge1['timestamp'].apply(lambda timestamp: datetime.fromtimestamp(timestamp).year)
df_merge1.drop('timestamp', axis=1, inplace=True)

# -------------- Create a Figure that shows us that shows us how the Ratigs are distriuted. ----------------#
# Get the data
data = df_merge1['rating'].value_counts().sort_index(ascending=False)

ratings_df = pd.DataFrame()
ratings_df['Mean_Rating'] = df_merge1.groupby('title')['rating'].mean().values
ratings_df['Num_Ratings'] = df_merge1.groupby('title')['rating'].count().values

genre_df = pd.DataFrame(df_merge1['genres'].str.split('|').tolist(), index=df_merge1['movieId']).stack()
genre_df = genre_df.reset_index([0, 'movieId'])
genre_df.columns = ['movieId', 'Genre']

def make_bar_chart(dataset, attribute, bar_color='#3498db', edge_color='#2980b9', title='Title', xlab='X', ylab='Y', sort_index=False):
    if sort_index == False:
        xs = dataset[attribute].value_counts().index
        ys = dataset[attribute].value_counts().values
    else:
        xs = dataset[attribute].value_counts().sort_index().index
        ys = dataset[attribute].value_counts().sort_index().values


    fig, ax = plt.subplots(figsize=(14, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel(xlab, fontsize=16, labelpad=20)
    ax.set_ylabel(ylab, fontsize=16, labelpad=20)

    plt.bar(x=xs, height=ys, color=bar_color, edgecolor=edge_color, linewidth=2)
    plt.xticks(rotation=45)

 # Merging the merge data earlier on with the df_imbd
df_merge3 = df_merge1.merge(df_imdb, on = "movieId" )

num_ratings = pd.DataFrame(df_merge3.groupby('movieId').count()['rating']).reset_index()
df_merge3 = pd.merge(left=df_merge3, right=num_ratings, on='movieId')
df_merge3.rename(columns={'rating_x': 'rating', 'rating_y': 'numRatings'}, inplace=True)

# pre_process the budget column

# remove commas
df_merge3['budget'] = df_merge3['budget'].str.replace(',', '')
# remove currency signs like "$" and "GBP"
df_merge3['budget'] = df_merge3['budget'].str.extract('(\d+)', expand=False)
#convert the feature into a float
df_merge3['budget'] = df_merge3['budget'].astype(float)
#remove nan values and replacing with 0
df_merge3['budget'] = df_merge3['budget'].replace(np.nan,0)
#convert the feature into an integer
df_merge3['budget'] = df_merge3['budget'].astype(int)

df_merge3['release_year'] = df_merge3.title.str.extract('(\(\d\d\d\d\))', expand=False)
df_merge3['release_year'] = df_merge3.release_year.str.extract('(\d\d\d\d)', expand=False)

data_1= df_merge3.drop_duplicates('movieId')

# Movies published by year:

years = []

for title in df_merge3['title']:
    year_subset = title[-5:-1]
    try: years.append(int(year_subset))
    except: years.append(9999)

df_merge3['moviePubYear'] = years
print('The Number of Movies Published each year:',len(df_merge3[df_merge3['moviePubYear'] == 9999]))

def make_histogram(dataset, attribute, bins=25, bar_color='#3498db', edge_color='#2980b9', title='Title', xlab='X', ylab='Y', sort_index=False):
    if attribute == 'moviePubYear':
        dataset = dataset[dataset['moviePubYear'] != 9999]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel(xlab, fontsize=16, labelpad=20)
    #ax.set_yticklabels([yticklabels(item, 'M') for item in ax.get_yticks()])
    ax.set_ylabel(ylab, fontsize=16, labelpad=20)

    plt.hist(dataset[attribute], bins=bins, color=bar_color, ec=edge_color, linewidth=2)

    plt.xticks(rotation=45)

def view_profile(name, email, age, gender):
    st.write("View Profile:")
    st.write(f"Name: {name}")
    st.write(f"Email: {email}")
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")

def edit_profile():
    st.write("Edit Profile:")
    name = st.text_input("Name", "John Doe")
    email = st.text_input("Email", "john@example.com")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.radio("Gender", ["Male", "Female"])
    return name, email, age, gender



# ------------------------------ CODE FOR THE FIGURES ENDS HERE ------------------------------------#

# App declaration
def main():
    html_template = """
        <div style="background-color:black;padding:10px;border-radius:10px;margin:10px;">
        <h1 style="color:green;text-align:center;">EDSA Movie Recommendation Challenge</h1>
        <h2 style="color:white;text-align:center;">UNSUPERVISED LEARNING PREDICT - TEAM FM3</h2>
        </div>
        """

    title_template = """
        <div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:20px;">
        <h1 style="color:white;text-align:center;">UNSUPERVISED LEARNING PREDICT</h1>
        <h2 style="color:white;text-align:center;">TEAM FM3</h2>
        <h3 style="color:white;text-align:center;">Nkosingiphile Sefodi</h3>
        <h3 style="color:white;text-align:center;">Zinhle Mjwara</h3>
        <h3 style="color:white;text-align:center;">Jean Rabothata</h3>
        <h3 style="color:white;text-align:center;">Thabang Ntuli </h3>
        <h3 style="color:white;text-align:center;">Anele</h3>
        <h3 style="color:white;text-align:center;">Nompumeza</h3>
        </div>
        """

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home","Recommender System","About","EDA", "Solution Overview", "About Us"]

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
    if page_selection == "Home":
        st.markdown(html_template.format('royalblue', 'white'), unsafe_allow_html=True)
        st.image('resources/imgs/Home.PNG', use_column_width=True)
        # st.markdown(title_template, unsafe_allow_html=True)

    if page_selection == "About":
        # markup(page_selection)
        st.write("### Oveview: Flex your Unsupervised Learning skills to generate movie recommendations")

        # You can read a markdown file from supporting resources folder
        # if st.checkbox("Introduction"):
        st.subheader("Introduction to Unsupervised Learning Predict")
        st.write(
            """In today’s technology driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.""")
        st.write(
            """With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.""")
        st.write(
            """Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity.""")

        # if st.checkbox("Problem Statement"):
        st.subheader("Problem Statement of the Unsupervised Learning Predict")
        st.write("Build recommender systems to recommend a movie")

        # if st.checkbox("Data"):
        st.subheader("Data Overview")
        st.write(
            """This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!""")

        st.write(
            """For this Predict, we'll be using a special version of the MovieLens dataset which has enriched with additional data, and resampled for fair evaluation purposes.""")

        st.write("""### Source:""")
        st.write(
            """The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB""")

        st.write("""### Supplied Files:
        genome_scores.csv - a score mapping the strength between movies and tag-related properties. Read more here

        genome_tags.csv - user assigned tags for genome-related scores

        imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.

        links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.

        sample_submission.csv - Sample of the submission format for the hackathon.

        tags.csv - User assigned for the movies within the dataset.

        test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.

        train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.""")

        # st.subheader("Raw Twitter data and label")
        # if st.checkbox('Show raw data'): # data is hidden if box is unchecked
        #     st.write(raw[['sentiment', 'message']]) # will write the df to the page
    if page_selection == "EDA":
        st.header("Movie Recommender System Datasets explorer")
        st.markdown("""EDA is the Exploratory Data Analsis used to gain insight into the dataset""")
        st.sidebar.header("Select Visuals to Display")
        #st.sidebar.subheader("Available Visuals obtained from the sections below:")
        all_cols = df_merge1.columns.values
        numeric_cols = df_merge1.columns.values
        obj_cols = df_merge1.columns.values



        if st.sidebar.checkbox("Visuals on Ratings"):
            if st.checkbox("Ratings count by year"):
                fig, ax = plt.subplots(1, 1, figsize = (12, 6))
                ax1 = df_merge1.groupby('rating_year')['rating'].count().plot(kind='bar', title='Ratings by year')
                st.write(fig)


            if st.checkbox("How ratings are distributed: histogram"):
                f = px.histogram(df_merge1["rating"], x="rating", nbins=10, title="The Distribution of the Movie Ratings")
                f.update_xaxes(title="Ratings")
                f.update_yaxes(title="Number of Movies per rating")
                st.plotly_chart(f)
            if st.checkbox("How ratings are distributed: scatter plot"):
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title('Rating vs. Number of Ratings', fontsize=24, pad=20)
                ax.set_xlabel('Rating', fontsize=16, labelpad=20)
                ax.set_ylabel('Number of Ratings', fontsize=16, labelpad=20)

                plt.scatter(ratings_df['Mean_Rating'], ratings_df['Num_Ratings'], alpha=0.5, color='green')
                st.pyplot(fig)

        if st.sidebar.checkbox("Visuals on Genres"):
            st.info("The number of movie per genre")
            fig=make_bar_chart(genre_df, 'Genre', title='Most Popular Movie Genres', xlab='Genre', ylab='Counts')
            st.pyplot(fig)

        if st.sidebar.checkbox("Movie published"):
            st.info("Movies published by year")
            st.pyplot(make_histogram(df_merge3, 'moviePubYear', title='Movies Published per Year', xlab='Year', ylab='Counts'))




    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Our winning approach")

        st.markdown("""`Recommender Systems` are one of the primary methods companies are using to enhance the user experience for customers 
        on their platforms. Major companies like Netflix, HBO, Amazon, Youtube and Facebook have `spent millions of dollars` to improve content 
        delivery and the user experience, in order to have better customer retention, better growth of their user base and superior content delivery 
        and visibility to attract more publishers for their platform. `All of these companies are using a Recommender System` to do this!
        """)

        st.subheader("Content Based Filtering VS Collaborative Filtering")
        st.markdown("There are two primary approaches to create a Recommender System. We tested both Content Based Filtering and Collaborative Filtering to find the best method that would make our system the most effective")
        image=Image.open("./resources/imgs/contentVScollab.png")
        st.image(image, use_column_width=True)

        st.info("**Content Based Filtering**")
        st.markdown("""The Content based filtering method is based on the descriptive data available for a particular item or product. This method 
        will look at items that are similar in their description and context establish recommendations based on that. The user’s prior preferences 
        are taken into consideration to find products that match their historic preferences. If for example, a user likes to watch Toy Story, then 
        they will be recommended movies that are very similar to it, like the rest of the Toy Story series and other Pixar movies
        """)

        st.info("**Collaborative Filtering**")
        st.markdown("""The Collaborative Filtering technique filters recommendations for a user based on the reactions by other similar users, 
        resulting in recommendations gathered and generated by many users instead of just one.
        This method will use a large user base and look for subsets of them with similar preferences to a particular user. It will base the recommendations 
        on the similarities in their preferences and provide a wider range of content to the user.<br> <br>
        Collaborative Filtering is a very popular method to use for movie recommendation systems which allowed for easy implementation due to an abundance of 
        information on it. It also enables easy maintenance and continued modification due to its popularity and relative ease of use. It is also lighter on performance as apposed to the 
        Content Based Filter resulting in increased performance.
        """, unsafe_allow_html=True)

        st.success("**Winner: Collaborative Filtering!**")
        st.markdown("""It is a better method to use for our recommender system since to it can conduct feature learning on its own, which will enable it to learn 
        which features are best to use and it will only grow more useful in time. Collaborative filtering users other user ratings to create decisions and adapts 
        to the users interests which will inevitably change over time. Collaborative Filtering is vastly superior to the rigid and restrictive content based filtering method.
        """, unsafe_allow_html=True)

        st.title("**Why you should use our Recommender Sytem**")
        st.markdown("""We created our recommender system with a focus on being user friendly and providing the best experience to the user. We want the 
        user to be exposed to familiar and new content alike and with using our custom algorithm we have achieved this. Our Recommender System will not 
        only attract more customers to the platform but also have great customer retention as our system only gets more effective the more it is used. <br> <br> 
        Our system is easy to use and implement and will enable any company using it to rival all other similar content driving platforms. """, unsafe_allow_html=True)

    elif page_selection == "User Profile":
        st.title("User Profile")
        st.sidebar.title("User Profile Management")

        # Sidebar menu for profile management
        profile_options = ["View Profile", "Edit Profile", "Preferences"]
        selected_option = st.sidebar.radio("Select Option", profile_options)

        name, email, age, gender = "John Doe", "john@example.com", 30, "Male"

        if selected_option == "View Profile":
            st.write("View Profile:")
            # Placeholder for viewing user profile
            view_profile(name, email, age, gender)

        elif selected_option == "Edit Profile":
            st.write("Edit Profile:")
            name, email, age, gender = edit_profile()
            # Placeholder for editing user profile
            if st.button("Update Profile"):
                st.success("Profile updated successfully!")
                view_profile(name, email, age, gender)


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

        # Define owners' information
        owners_info = [
            {"name": "Zinhle Mjwara", "role": "Role 1", "image_url": "resources/imgs/download.jpeg"},
            {"name": "Nkosingiphile Sefodi", "role": "Role 2", "image_url": "resources/imgs/butcher.jpg"},
            {"name": "Jean Rabothata", "role": "Role 3", "image_url": "resources/imgs/download.jpeg"},
            {"name": "Anele ", "role": "Role 4", "image_url": "resources/imgs/download.jpeg"},
            {"name": "Thabang Ntuli", "role": "Role 5", "image_url": "resources/imgs/download.jpeg"},
            {"name": "Nompumeza", "role": "Role 6", "image_url": "resources/imgs/download.jpeg"}
        ]

        # Create columns layout for displaying images and text
        col_count = 3  # Number of columns
        cols = st.columns(col_count)

        for i, owner_info in enumerate(owners_info):
            with cols[i % col_count]:
                st.image(owner_info["image_url"], use_column_width=True)
                st.write(owner_info["name"])
                st.write(owner_info["role"])


    if page_selection == "About Us":
        st.info('**The legends who made it happen**')
        st.write('**Nkosingiphile Sefodi** - Programming and Modeling')
        st.write('**Zinhle Mjwara** - Data Analysis and Documentation')
        st.write('**Jean Rabothata** - Data Analysis and Documentation')
        st.write('**Thabang Ntuli** - Data Analysis and Documentation')
        st.write('**Anele** - Data Analysis and Documentation')
        st.write('**Nompumezo** - Data Analysis and Documentation')
        st.image('resources/imgs/EDSA_logo.png',use_column_width=True)

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.



if __name__ == '__main__':
    main()
