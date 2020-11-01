import numpy as np
import yaml
import pickle
import os

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import ValidationError
from wtforms.validators import InputRequired, Email, Length
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
model = pickle.load(open('model_GB.pkl', 'rb'))

ENV = 'dev'

def get_config(fname):
    '''
    Creates connection to yaml file which holds the DB user and pass
    '''
    with open(fname) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg

if ENV == 'prod':

    cfg = get_config('config.yml')
    connection = cfg['connection'][ENV]
    app.config['SECRET_KEY'] = connection['secret_key']
    app.debug = True
    app.config[connection['username']] = connection['password']

    app.config['TESTING'] = False
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 25
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL__USE_SSL'] = False
    app.config['MAIL_USERNAME'] = connection['mail_user']
    app.config['MAIL_PASSWORD'] = connection['mail_pass']
    app.config['MAIL_DEFAULT_SENDER'] = 'mail@syndicate.com'
    app.config['MAIL_MAX_EMAILS'] = None
    app.config['MAIL_ASCII_ATTACHMENTS'] = False

else:
    app.debug = False
    app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
    app.config['MAIL_SERVER'] = os.environ['MAIL_SERVER']
    app.config['MAIL_PORT'] = 25
    app.config['MAIL_USE_TLS'] = False
    app.config['MAIL__USE_SSL'] = False
    app.config['MAIL_USERNAME'] = os.environ['MAIL_USERNAME']
    app.config['MAIL_PASSWORD'] = os.environ['MAIL_PASSWORD']

    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

mail = Mail(app)
Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('UserName', validators = [InputRequired(), Length(min = 4, max = 15)])
    password = PasswordField('Password', validators = [InputRequired(), Length(min = 8, max = 80)])
    remember = BooleanField('Remember Me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators = [InputRequired(), Email(message = 'Invalid Email'), Length(max = 50)])
    username = StringField('UserName', validators = [InputRequired(), Length(min = 4, max = 15)])
    password = PasswordField('Password', validators = [InputRequired(), Length(min = 8, max = 80)])

    def validate_username(self, username):
        '''
        Raises a validation error if a user tries to register using an existing username
        '''
        user = User.query.filter_by(username = username.data).first()
        if user:
            raise ValidationError('Username Taken')

    def validate_email(self, email):
        '''
        Raises a validation error if a user tries to register using an existing email
        '''
        user = User.query.filter_by(email = email.data).first()
        if user:
            raise ValidationError('Email Taken')

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/error/')
def error():
    return render_template('error.html')

@app.route('/login_error/')
def login_error():
    return render_template('login_error.html')

@app.route('/login/',methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username = form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember = form.remember.data)
                flash('Account Created For {}!'.format(form.username.data))
                return redirect(url_for('model_page'))
        else:
            return redirect(url_for('login_error'))

    return render_template('login.html', form=form)

@app.route('/signup/', methods = ['GET','POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method = 'sha256') # sha256 will generate a hash which is 80 chars long
        new_user = User(username = form.username.data, email = form.email.data, password = hashed_password)
        db.session.add(new_user)
        db.session.commit()

        # send congrat email for registering
        # try:
        msg = Message(subject = 'Welcome {}'.format(form.username.data), sender = app.config.get("MAIL_USERNAME"), recipients = [str(form.email.data)], body = 'Congratulations you have signed up and your account has been created!')
        mail.send(msg)

        return redirect(url_for('login'))
    else:
        return render_template('signup.html', form = form, message= 'Username / Email Already Exists')
        # return '<h1>' + form.email.data + ' ' + form.username.data + ' ' + form.password.data + '<h1>'
    return render_template('signup.html', form = form)

@app.route('/logout/')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/generic/',methods=['GET', 'POST'])
def generic():
    return render_template('generic.html')

@app.route('/account/',methods=['GET', 'POST'])
@login_required
def account():
    return render_template('account.html')

@app.route('/model_page/', methods = ['GET','POST'])
@login_required
def model_page():
    return render_template('model_page.html')

@app.route('/predict_model', methods=['GET', 'POST'])
def predict_model():
    int_features = [int(x) for x in request.form.values()]
    print('pass1')
    final_features = [np.array(int_features)]
    print('pass2')
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    map_dict = {1 : 'DT Toronto', 3 : 'North York', 4 : 'Scarborough', 6 : 'Etobicoke'}
    output = map_dict[output]
    print('pass3')
    return render_template('model_page.html', prediction_text = 'The Crime Occurred in Burrough : {}'.format(output))

if __name__ == "__main__":
    if ENV == 'prod':
        app.run()
    else:
        app.run(debug=True)
