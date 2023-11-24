from flask import render_template, flash, redirect, url_for
from flask_login import current_user, login_user, logout_user, login_required
from app.forms import LoginForm, RegistrationForm,  UploadImageForm
from app import app, db, ImageCaptioning
from app.models import User
from flask import request
from werkzeug.urls import url_parse
from werkzeug.utils import secure_filename
import os

@app.route('/')
@app.route('/index')
@login_required
def index():
    form = UploadImageForm()
    return render_template('index.html',title='Home',form=form)

@app.route('/login', methods = ['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user =  User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username of password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_parse('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In',form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/uploadimage', methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        print(request.files)
        if 'fileupload' not in request.files:
            flash('No file Part. Please upload the file')
            return redirect(url_parse('index'))

        #file is there
        file = request.files['fileupload']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_parse('index'))
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            caption =ImageCaptioning.predict_user_images(filename)
            flash(caption)
            print(caption)
            return redirect(url_parse('index'))
        else:
            flash('Invalid file type. Allowed file types are: png, jpg, jpeg, gif')
            return redirect(url_parse('index'))

