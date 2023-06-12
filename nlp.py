import nltk
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string

# Définir la liste des mots vides (stop words) en français
stop_words = set(stopwords.words('french'))

# Définir le stemmer pour la racinisation (stemming)
stemmer = SnowballStemmer('french')

def preprocess_text(text):
    if isinstance(text, str): # Vérifier si le texte est une chaîne de caractères valide
        # Convertir en minuscules
        text = text.lower()

        # Supprimer la ponctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Supprimer les caractères spéciaux et les chiffres
        text = re.sub(r'[^a-z ]', '', text)

        # Tokenization des mots
        words = nltk.word_tokenize(text)

        # Supprimer les mots vides (stop words)
        words = [word for word in words if word not in stop_words]

        # Appliquer la racinisation (stemming)
        words = [stemmer.stem(word) for word in words]

        # Rejoindre les mots en une seule chaîne de caractères
        processed_text = ' '.join(words)

        return processed_text
    else:
        return '' # Retourner une chaîne de caractères vide pour les valeurs non valides