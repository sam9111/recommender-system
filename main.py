import streamlit as st
import pandas as pd
from bertopic_model import *
from scrapy_fetch import fetch_data
import pandas
from recommender import *
import os


st.set_page_config(layout="wide")


def fetch_latest():
    with st.spinner('Fetching latest articles and creating topics...'):

        fetch_data()

        # Load data
        df = pd.read_csv('./data/data.csv', delimiter=',', header=0)

        # filter out all rows that are from 2022

        date_df = df[df['published_date'].str.contains('2022|2021')]

        df = df.drop(date_df.index)

        df = df.dropna()

        df = df.drop_duplicates(subset=['processed_title'])

        df = df.sample(frac=1).reset_index(drop=True)

        df.to_csv('./data/data.csv')

        # Tokenize data

        docs = [str(i) for i in df['processed_title'].tolist()]

        run(docs)

        print("Topics created")


def main():

    if not os.path.exists('./data/data.csv') or not os.path.exists('./data/embeddings.pkl') or not os.path.exists('./data/topic_model'):
        fetch_latest()
    df = pd.read_csv('./data/data.csv', delimiter=',', header=0)

    topics = get_topics()
    topic_info = get_topic_info()

    st.title("News Articles and Topics")

    with st.sidebar:
        st.button("Fetch Latest", on_click=fetch_latest)
        st.write("Topic Info")
        st.write(topic_info[['CustomName', 'Topic', 'Count']])

    # Concatenate data and topics
    topics_df = pd.DataFrame({'topic': topics})
    df_with_topics = pd.concat([df, topics_df], axis=1)
    df_with_topics['topic'] = df_with_topics['topic'].astype(str)

    # display the dataframe

    # sort data by published date
    data = df_with_topics
    # Set page size and number of pages
    page_size = 10
    num_pages = int(len(data) / page_size)

    col1, col2 = st.columns([1, 2], gap="large")

    # Create pagination controls
    page_num = st.number_input(
        "Page", value=1, min_value=1, max_value=num_pages, step=1)
    offset = (page_num - 1) * page_size

    # Create a container for the table
    container = st.container()

    # Create columns for each data field

    # Iterate over data rows and display the current page
    for i in range(offset, offset + page_size):
        if i >= len(data):
            break
        row = data.iloc[i]
        with container:
            col1.write(f"**{row['title'].strip()}**")
            col1.write(row['link'])
            col1.write(row['published_date'])
            col1.write("Belongs to topic " +
                       row['topic'])

            if col1.button("View", key=i):
                article = data.iloc[i]

                recommended = recommend_articles(
                    article, data, load_embeddings())
                # remove the article itself from the recommendations
                recommended = recommended[recommended['link']
                                          != article['link']]
                col2._iframe(article.link, height=500, scrolling=True)
                col2.write("**Recommended articles:**")
                for index, row in recommended.iterrows():
                    col2.write(f"**{row['title'].strip()}**")
                    col2.write(row['link'])
                    col2.write(row['published_date'])
                    col2.write("Belongs to topic " +
                               row['topic'])
                    col2.divider()

            col1.divider()


if __name__ == '__main__':
    main()
