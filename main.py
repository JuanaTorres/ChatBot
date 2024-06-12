import warnings
warnings.filterwarnings("ignore")

import pandas as pd

#Read data from Google Sheet
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vRj7Lv8YMJWVd91cWpVeCd_KbCVLbG3Psk5DitPm3ShuKiAtUvfqn0gMtxDlteOp-RUssvcccRlnw-v/pub?output=tsv", sep="\t")

#print(df)
#Transform dataset: Opinion-Type
good_df = df[['Positivos']]
good_df['Opinion'] = "POSITIVE"

bad_df = df[['Negativos']]
bad_df.columns = ['Positivos']
bad_df['Opinion'] = "NEGATIVE"

df_op = pd.concat([good_df,bad_df])
df_op.columns = ['Opinion','Type']

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
import unicodedata

nltk.download('stopwords')
stemmer = SnowballStemmer('spanish')
nltk.download('punkt')

stop_words = set(stopwords.words('spanish'))
stop_words = stop_words.union(set(['vide', 'jueg', 'videojueg', 'me', 'le', 'da', 'mi', 'su', 'ha', 'he', 'ya', 'un', 'una', 'es','del', 'las', 'los', 'en', 'que', 'y', 'la','de']))

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def fast_preproc(text):
  text = text.lower()
  text = ''.join(c for c in text if not c.isdigit())
  text = ''.join(c for c in text if c not in punctuation)
  text = remove_accents(text)
  words = word_tokenize(text)
  words = [stemmer.stem(word) for word in words]
  words = [word for word in words if not word in stop_words]
  try:
    text = " ".join(str(word) for word in words)
  except Exception as e:
    print(e)
    pass
  return text

df_op['Opinion'] = df_op['Opinion'].astype(str)

df_op = df_op.assign(
    TextPreproc=lambda df: df_op.Opinion.apply(fast_preproc)
)
#Split dataset
X = df_op['TextPreproc']
Y = df_op['Type']

#print("X:")
#print(X[0:5])
#print("\nY:")
#print(Y[0:5])
'''
#from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_df=0.5)

#Tokenize and build vocabulary
vec.fit(X)
#print("\nVocabulary:\n")
#print(vec.vocabulary_)

#Encode documents
trans_text_train = vec.transform(X)

#Print Document-Term Matrix
df = pd.DataFrame(trans_text_train.toarray(), columns=vec.get_feature_names_out())
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.1)
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#Classic machine-learning algorithms
#classifier = LogisticRegression(random_state=0,penalty='l2',solver='lbfgs')
classifier = DecisionTreeClassifier(random_state=0)
#classifier = RandomForestClassifier(random_state=0)

#Machine learning algorithms used in Text Classification
#classifier = svm.SVC()
#classifier = MultinomialNB()

classifier.fit(X_train, Y_train)

from sklearn.metrics import classification_report,confusion_matrix

#Predice para el conjunto de testeo.
y_pred = classifier.predict(X_test)

print("\nMatriz de confusión:\n")
print(confusion_matrix(Y_test,y_pred))

print("\nEstadisticas del clasificador:\n")
print(classification_report(Y_test,y_pred))

def clasificarComentario(comentario):
   new_opinion = [comentario]
    #new_opinion = ["¡Guau! Tuve un día terrible en el trabajo, pensé que este juego sería un ANTIESTRÉS para mí... Resulta que es más bien un ANTIESTRÉS... Quiero decir, ¡la cantidad de estrés que genera este juego está fuera de serie! Sin embargo, ¡Recomendaría este juego a todos y cada uno de ustedes! Pero cuidado, NO JUGUEN SI SU ESTADO DE ÁNIMO  AHORA MISMO... Saludos."]
   Xt_new = [fast_preproc(str(new_opinion))]
   trans_new_doc = vec.transform(Xt_new) #Use same TfIdfVectorizer
   print("\nPredicted result: " + str(classifier.predict(trans_new_doc)))
   return str(classifier.predict(trans_new_doc))
import os
'''
import flask
from flask import send_from_directory, request
from twilio.rest import Client

app = flask.Flask(__name__)

account_sid = 'aount'
auth_token = 'token'
client = Client(account_sid, auth_token)

@app.route('/')
@app.route('/home')
def home():
    client.messages.create(
        from_='whatsapp:+14155238886',
        body="Hola, porfavor escribe tu opinón sobre nuestro video juego",
        to='whatsapp:+573185578754')
    return "Hello World"

def sendMessage(message):
  #tipoComentario=clasificarComentario(message)
  if("tipoComentario"==['NEGATIVE']):
#change nums
      client.messages.create( from_='whatsapp:numtwilio',body='Lo sentimos, mejoraremos el juego para la proxima.', to='whatsapp:num')
  else:
      client.messages.create(from_='whatsapp:numtwilio', body='Gracias por su comentario.',
                             to='whatsapp:num')
#

@app.route('/what', methods=['POST'])
def what():
    #processed_request=process_request(request.values.get('Body'))

    print(request.get_data())
    message = request.form['Body']
    senderId = request.form['From'].split('+')[1]
    print(f'Message --> {message}')
    print(f'Sender id --> {senderId}')
    res = sendMessage(message)
    print(f'This is the response --> {res}')
    return '200'

from flask_ngrok import run_with_ngrok
run_with_ngrok(app)
if __name__ == "__main__":
    app.run()