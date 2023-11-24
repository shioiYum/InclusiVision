import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config():
    SECRET_KEY = os.environ.get('SECRET KEY') or 'you'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATION = False
    UPLOAD_FOLDER_IMAGE = os.environ.get('UPLOAD_FOLDER_IMAGE') or os.path.join(basedir, 'Media', 'Images')
    UPLOAD_FOLDER_VIDEO = os.environ.get('UPLOAD_FOLDER_VIDEO') or os.path.join(basedir, 'Media', 'video')
