import numpy as np
import yaml
import pickle
import os

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
model = pickle.load(open('model_GB.sav', 'rb'))
# app.config['SECRET_KEY'] = 'Thisissupposedtobesecret'
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']

ENV = 'prod'

def get_config(fname):
    '''
    Creates connection to yaml file which holds the DB user and pass
    '''
    with open(fname) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg

cfg = get_config('config.yml')
connection = cfg['connection'][ENV]

if ENV == 'dev':
    app.debug = True
    app.config[connection['username']] = connection['password']
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
    # app.config[connection['username']] = os.environ['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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



@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/login/',methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username = form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember = form.remember.data)
                return redirect(url_for('element'))

            return '<h1> Invalid Username or Password </h1>'

        # return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)

@app.route('/signup/', methods = ['GET','POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method = 'sha256') # sha256 will generate a hash which is 80 chars long
        new_user = User(username = form.username.data, email = form.email.data, password = hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
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

@app.route('/element/',methods=['GET', 'POST'])
@login_required
def element():
    return render_template('elements.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    map_dict = {1 : 'DT Toronto' , 3 : 'North York', 4 : 'Scarborough', 6 : 'Etobicoke'}
    output = map_dict[output]
    return render_template('index.html', prediction_text='The Crime Occured In Burrough : {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]


    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
