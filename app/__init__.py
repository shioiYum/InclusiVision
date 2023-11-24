from flask import Flask
from app.config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

login = LoginManager(app)
login.login_view = 'login'

vgg_model = VGG16()
    # restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

from keras.models import load_model
basedir = os.path.abspath(os.path.dirname(__file__))
img_model = load_model(basedir + '/Models/ImageCaptioning.h5')
with open(basedir + '/Models/ImageTokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

from app import routes, models, vgg_model, loaded_tokenizer, img_model
