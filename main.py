import streamlit as st
import pandas as pd
from bertopic_model import *
from scrapy_fetch import fetch_data
import pandas
from recommender import *

st.set_page_config(layout="wide")

def main():
    st.title("News Articles and Topics")

    # with st.spinner('Fetching data...'):
    #     fetch_data()

    with st.spinner('Running BERTopic model...'):
     
        # Load data
        df = pd.read_csv('data.csv', delimiter=',', header=0)

        # Tokenize data

        docs = [str(i) for i in df['title'].tolist()]

        # run(docs)

        topics=get_topics()

        # Concatenate data and topics
        topics_df = pd.DataFrame({'topic': topics})
        df_with_topics = pd.concat([df, topics_df], axis=1)
        df_with_topics['topic'] = df_with_topics['topic'].astype(str)

    # display the dataframe

    data=df_with_topics
        # Set page size and number of pages
    page_size = 10
    num_pages = int(len(data) / page_size)

    # Create pagination controls
    page_num = st.number_input("Page", value=1, min_value=1, max_value=num_pages, step=1)
    offset = (page_num - 1) * page_size

    # Create a container for the table
    container = st.container()

    # Create columns for each data field
    col1, col2 = st.columns([1,2])

    # Iterate over data rows and display the current page
    for i in range(offset, offset + page_size):
        if i >= len(data):
            break
        row = data.iloc[i]
        with container:
            col1.write(f"**{row['title']}**")
            col1.write(row['link'])
            col1.write(row['published_date'])
            col1.write("Belongs to topic: "+ row['topic'])
            if col1.button("View", key=i):
                article=data.iloc[i]
                recommend_articles(article,data,)
                col2._iframe(article.link, height=500, scrolling=True)


            col1.divider()
            

    # Create pagination buttons
    if page_num > 1:
        st.button("Previous", key="prev")
    if page_num < num_pages:
        st.button("Next", key="next")



if __name__ == '__main__':
    main()
