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
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import toml
import time

from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu 
from PIL import Image

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# This command allows the app to use wide mode of the screen.
# st. set_page_config(layout="wide")


# Use local CSS to sort the styling of the contact form
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")


# Function to access the json files of the lottie animations
def load_lottieurl(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Visualisations","Data Information", "Solution Overview", "About Us!"]
    
    st.sidebar.image("images/Astro_Coders1.png", use_column_width=True)

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
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    
                    st.success("##### We think you'll like:")
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
                    
                    st.success("##### We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        
        st.subheader("Our winning approach")

        st.write("""One of the main strategies used by businesses to improve the user experience for users on their platforms is the use of recommender systems.
         Major corporations like Netflix, HBO, Amazon, Youtube, and Facebook have invested millions of dollars to enhance user experience and content delivery 
         in order to increase user growth, boost customer retention, and draw more publishers to their platforms. 
         To accomplish this, each of these businesses uses a recommender system.
        """)

        st.subheader("Content Based Filtering VS Collaborative Filtering")
        st.write("There are primarily two methods for developing a recommender system. We experimented with both collaborative filtering and content-based filtering to see which approach would produce the greatest results for our system.")
        
        st.image("images/contentVScollab2.png")

        st.info("##### Content-based Filtering")
        st.write("""Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback. 
        This approach will examine items with comparable descriptions and make recommendations based on that. 
        Finding products that match the user's historical tastes takes into account their earlier preferences. 
        One advantage of Content-based filtering is that the model does not need any data about other users, since the recommendations are specific to this user. 
        This makes it easier to scale to a large number of users. The model can capture the specific interests of a user, and can recommend niche items that very few other users are interested in.
        """)

        st.info("##### Collaborative-based Filtering")
        st.write("""Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users. 
        It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user. 
        It will give the user access to a greater variety of content and base recommendations on how similar their interests are.
        Due to its widespread use and comparatively simple operation, it also makes straightforward maintenance 
        and ongoing customization possible. Additionally, it performs better than the Content Based Filter because it uses less processing power.
        """)

        st.success("##### Winner: Collaborative-based Filtering")
        st.write("""Our best filtering method goes to the Collaborative-based method for a number of reasons. The primary advantage of collaborative filtering is that users can get broader exposure to many different movies , which creates possibilities to encourage them towards continual usage of the platform. 
        Collaborative filtering gives suggestions because most of the unknown buyers have a similar taste to you. Still, in Content-based, you will get the recommendations of movies based on their features. In contrast to Collaborative filtering, new new movies are suggested without any specifications to the users.
        The collaborative filtering model can help users discover new interests and although the ML system might not know the users interest in a given movie, the model might still recommend it because similar users are interested in that movie. 
        On the other hand, A Content-based model can only make recommendations based on the existing interests of the user and the model hence only has limited ability to expand on the users existing interests.
        """)
        

        st.info("##### Why we believe our Recommender System is the best in the market! ")
        st.write("""We created our recommender system with a focus on being user friendly and providing the best experience to the user. We want the 
        user to be exposed to familiar and new content alike and with using our custom algorithm we have achieved this. Our Recommender System will not 
        only attract more customers to the platform due to it's user-friendliness but also have great customer retention which that implies one thing,
        more customers will subscribe to the platform to get the best movie recommendations, thus leading to a higher profit margin for the company.
        And last but not least, Our system will also enable any company using it to be at the top of the game against other platforms driving similar content. """)


    if page_selection == "Visualisations":
        st.title("Visualisations")
        st.write("##### For this section we will explore the distribution of the data from genres, user preferences, ratings etc.")
        st.write("---")

        st.write("##### Bar graph showing Most popular tags")
        st.image("images/Tags.png")
        st.write("""
		#### Quick overview of the data above:
* There is a strong imbalance amongst the tags found on the dataset.
* Based on the information displayed by our bar graph, we can see that the top 3 leading tags include Sci-Fi, Atmospheric as well as Action.
* Only a few distribution is shown under the Funny, Visually appealing and Dystopia.
* This tells us what the tastes of most users are. Some actionable things we can take from this is the types of movies we can offer on the platform that are in-line with our users tastes
                    
				""")
        st.write("---")

        st.write("##### Bar graph showing Top Rated Movies")
        st.image("images/TopMovies.png")
        st.write("""
		#### Quick overview of the data above:
Here we see the highest rated movies are older moves, some from the 90s and 2000s.Special events could be had around these kind of movies like a “Star wars theme” week where we promote movies that are similar to star wars or theme the user interface like star wars or whatever the top rated movie for a certain user is. This could have a positive impact on the service’s brand loyalty

				""")
        st.write("---")

        st.write("##### Bar graph showing Number of Movies for each Rating")
        st.image("images/Ratings.png")
        st.write("""
		#### Quick overview of the data above:
Here we can talk about the fact that most of the ratings fall on the positive side so that means the users like the movies that are being offered. Further investigation can be done to produce more content that are similar or offer content that are similar to the highest rated ones to increase and maintain user attention
				""")
        
        

    if page_selection == "Data Information":
        st.title("Data Information")
        st.info("##### Data Overview")
        
        left_column, right_column = st.columns(2)
        with left_column:
            st.write('''This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. 
                The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, 
                and now you get to as well!.''')

        with right_column:
            # Loading the animation in the "Data Information" section.
             data = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_fasueuv1.json")
             st_lottie(data, height=300, key="coding")
        
        st.info("##### More Information about the data")

        source = st.checkbox("Data Source")
        if source:
            st.write("### Where did we get the data from?")
            logo = Image.open("images/IMBD.png")
            st.image("images/IMBD.png", use_column_width=True)
            st.write('''The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering 
            at the University of Minnesota. Additional movie content data was legally scraped from IMDB''')
            st.write("---")
        
        
        look = st.checkbox("Supplied Data")
        if look:
            st.write("### List of files")
            st.write("##### Given below is a list of files that were legally scraped from IMBD")
            st.write('''* genome_scores.csv - a score mapping the strength between movies and tag-related properties.
                        \n* genome_tags.csv - user assigned tags for genome-related scores
                        \n* imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.
                        \n* links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.
                        \n* sample_submission.csv - Sample of the submission format for the hackathon.
                        \n* tags.csv - User assigned for the movies within the dataset.
                        \n* test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.
                        \n* train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.
                    ''')
            st.write("---")
 
        
    # Building the "About Us!" page
    if page_selection == "About Us!":
        st.title("Meet the Team!")
        
        st.info("##### Astro Coders")
        left_column, right_column = st.columns(2)
        with left_column:
            st.write(""" 
                    An organization based in Southern Africa with a passion for problem solving through the use of our team's unique skill set.
                    Astro Coders, founded in 2020 during the outbreak of Covid19, is one of Africa's largest IT and business consulting firms.
                    We are insight-driven and outcome-driven to help accelerate returns on your IT and business investments. 
                    Through personal client relationships and the provision of industry and technological expertise to assist you in meeting the needs of your customers and citizens, it is our mission to establish confidence in all we do.
        
            """)

        with right_column:
            # Loading the animation in the "About!" section.
             contact_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_zqndq8mg.json")
             st_lottie(contact_animation, height=300, key="coding")


        # Details of the team
        with st.container():
            st.info("##### Connect with us!")
            st.caption("### Chad Brache")
            st.write("""Supervisor
                    \nchad.brache@astrocoders.co.za""")
            
            st.write("---")

            left_column, right_column = st.columns(2)
            with left_column:
                st.caption("### Ozzey Padayachee")
                st.write("""Chief Executive Officer
                    \nozzey.padayachee@astrocoders.co.za""")
                        
                st.caption("### Sinhle Nkambule")
                st.write("""Web App Developer
                    \nsinhle.nkambule@astrocoders.co.za""")

                st.caption("### Nhlanhla Ngwenya")
                st.write("""Machine Learning Engineer
                    \nnhlanhla.ngwenya@astrocoders.co.za""")
        
            with right_column:
                st.caption("### Promise Lamola")
                st.write("""Data Science Manager
                    \npromise.lamola@astrocoders.co.za""")

                st.caption("### Samuel Mnisi")
                st.write("""Senior Data Analyst
                    \nsamuel.mnisi@astrocoders.co.za""")					  

                st.caption("### Sinethemba Nongqoto")
                st.write("""Data Scientist
                    \nsinethemba.nongqoto@astrocoders.co.za""")

        with st.container():
                st.info("##### Contact us using the form below:")
                st.write("---")

                # Documention: https://formsubmit.co/
                contact_form =( """
                <form action="https://formsubmit.co/sinhlenkambule78@gmail.com" method="POST">
                    <input type="hidden" name="_captcha" value="false">
                    <input type="text" name="name" placeholder="Your name" required>
                    <input type="email" name="email" placeholder="Your email" required>
                    <textarea name="message" placeholder="Your message here" required></textarea>
                    <button type="submit">Send</button>
                </form>
                """)
        
        
        left_column, right_column = st.columns(2)
        with right_column:
            
            # Loading the animation in the "Contact form" section.
            contact_animation = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_tfemgwhi.json")
            st_lottie(contact_animation, height=300, key="coding2")
        
        
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        
        st.title("Enjoy your movies! :wave:")

        st.write("---")
	

if __name__ == '__main__':
    main()
