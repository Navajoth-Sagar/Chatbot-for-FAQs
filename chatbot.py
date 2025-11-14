import json
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# ----------------------------
# Load FAQs
# ----------------------------
with open("faqs.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# ----------------------------
# Preprocess text
# ----------------------------
def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

# ----------------------------
# Vectorize
# ----------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# ----------------------------
# Chatbot Response
# ----------------------------
def chatbot_response(user_input):

    user_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_processed])

    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
    index = similarity_scores.argmax()
    score = similarity_scores.max()

    if score < 0.25:
        return "Sorry, I could not understand your question. Please try again."

    return answers[index]


# ----------------------------
# Terminal Chat (Testing)
# ----------------------------
if __name__ == "__main__":
    print("FAQ Chatbot is now running! Type 'exit' to quit.\n")
    while True:
        user = input("You: ")
        if user.lower() == "exit":
            break
        print("Bot:", chatbot_response(user))
