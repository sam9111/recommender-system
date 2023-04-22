import torch
from transformers import BertModel, BertTokenizerFast
import pickle
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pickle

file_path = "./data/"

tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")


def tokenize_ta(text, return_tensors="pt", *args, **kwargs):
    return indic_tokenize.trivial_tokenize(text)


stopwords = ['அங்கு',
             'அங்கே',
             'அடுத்த',
             'அதனால்',
             'அதன்',
             'அதற்கு',
             'அதிக',
             'அதில்',
             'அது',
             'அதே',
             'அதை',
             'அந்த',
             'அந்தக்',
             'அந்தப்',
             'அன்று',
             'அல்லது',
             'அவன்',
             'அவரது',
             'அவர்',
             'அவர்கள்',
             'அவள்',
             'அவை',
             'ஆகிய',
             'ஆகியோர்',
             'ஆகும்',
             'இங்கு',
             'இங்கே',
             'இடத்தில்',
             'இடம்',
             'இதனால்',
             'இதனை',
             'இதன்',
             'இதற்கு',
             'இதில்',
             'இது',
             'இதை',
             'இந்த',
             'இந்தக்',
             'இந்தத்',
             'இந்தப்',
             'இன்னும்',
             'இப்போது',
             'இரு',
             'இருக்கும்',
             'இருந்த',
             'இருந்தது',
             'இருந்து',
             'இவர்',
             'இவை',
             'உன்',
             'உள்ள',
             'உள்ளது',
             'உள்ளன',
             'எந்த',
             'என',
             'எனக்',
             'எனக்கு',
             'எனப்படும்',
             'எனவும்',
             'எனவே',
             'எனினும்',
             'எனும்',
             'என்',
             'என்ன',
             'என்னும்',
             'என்பது',
             'என்பதை',
             'என்ற',
             'என்று',
             'என்றும்',
             'எல்லாம்',
             'ஏன்',
             'ஒரு',
             'ஒரே',
             'ஓர்',
             'கொண்ட',
             'கொண்டு',
             'கொள்ள',
             'சற்று',
             'சிறு',
             'சில',
             'சேர்ந்த',
             'தனது',
             'தன்',
             'தவிர',
             'தான்',
             'நான்',
             'நாம்',
             'நீ',
             'பற்றி',
             'பற்றிய',
             'பல',
             'பலரும்',
             'பல்வேறு',
             'பின்',
             'பின்னர்',
             'பிற',
             'பிறகு',
             'பெரும்',
             'பேர்',
             'போது',
             'போன்ற',
             'போல',
             'போல்',
             'மட்டுமே',
             'மட்டும்',
             'மற்ற',
             'மற்றும்',
             'மிக',
             'மிகவும்',
             'மீது',
             'முதல்',
             'முறை',
             'மேலும்',
             'மேல்',
             'யார்',
             'வந்த',
             'வந்து',
             'வரும்',
             'வரை',
             'வரையில்',
             'விட',
             'விட்டு',
             'வேண்டும்',
             'வேறு']

vectorizer_model = CountVectorizer(
    stop_words=stopwords, analyzer='word',
    tokenizer=tokenize_ta
)

topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    verbose=True,
    calculate_probabilities=False,
    embedding_model=model,
)


def embeddings(docs):

    model.eval()
    n_docs = len(docs)
    batch_size = 8
    embeds = torch.zeros((n_docs, model.config.hidden_size))
    for i in range(0, n_docs, batch_size):
        batch = docs[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.pooler_output
        embeds[i:i+batch_size] = batch_embeddings

    with open(file_path+"embeddings.pkl", "wb") as f:
        pickle.dump(embeds, f)


def load_embeddings():
    with open(file_path+"embeddings.pkl", "rb") as f:
        embeds = pickle.load(f)

        return embeds.detach().numpy()


def run(data):

    embeddings(data)

    embeds = load_embeddings()

    trained_model = topic_model.fit(data, embeds)

    new_topics = trained_model.reduce_outliers(
        data, trained_model.topics_, strategy="c-tf-idf")
    trained_model.update_topics(
        data, topics=new_topics, vectorizer_model=vectorizer_model)

    trained_model.save(file_path+"topic_model")


def get_topic_info():
    trained_model = BERTopic.load(file_path+"topic_model")
    topic_labels = trained_model.generate_topic_labels(
        separator=", ", topic_prefix=False)
    trained_model.set_topic_labels(topic_labels)
    trained_model.save(file_path+"topic_model")
    return trained_model.get_topic_info()


def get_topics():
    trained_model = BERTopic.load(file_path+"topic_model")
    return trained_model.topics_


def get_topic_labels():
    trained_model = BERTopic.load(file_path+"topic_model")
    return trained_model.custom_labels_
