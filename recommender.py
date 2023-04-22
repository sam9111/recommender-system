

def recommend_articles(article, df, embeds_np):
    # find all documents with same topic as sample article and get their embeddings from embeds_np

    same_topic_docs = df[df['topic'] == article['topic']]

    same_topic_docs_embeddings = embeds_np[same_topic_docs.index]

    from sklearn.metrics.pairwise import cosine_similarity

    article_embedding = embeds_np[article.name].reshape(1, -1)

    cosine_similarities = cosine_similarity(article_embedding, same_topic_docs_embeddings).flatten()

    # find top 5 similar documents excluding the sample article

    similar_docs = same_topic_docs.iloc[cosine_similarities.argsort()[:-7:-1]]

    return similar_docs
