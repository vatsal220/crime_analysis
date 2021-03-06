import numpy as np
import yaml
import pickle
import os

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import ValidationError, DataRequired, EqualTo
from wtforms.validators import InputRequired, Email, Length
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer


app = Flask(__name__)
model = pickle.load(open('model_GB.pkl', 'rb'))

ENV = 'prod'

def get_config(fname):
    '''
    Creates connection to yaml file which holds the DB user and pass
    '''
    with open(fname) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg

if ENV == 'dev':

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

    def get_reset_token(self, expires_seconds = 1800):
        s = Serializer(app.config['SECRET_KEY'], expires_seconds)
        return s.dumps({'user_id' : self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return user.query.get(user_id)


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

class UpdateAccountForm(FlaskForm):
    email = StringField('email', validators = [InputRequired(), Email(message = 'Invalid Email'), Length(max = 50)])
    username = StringField('UserName', validators = [InputRequired(), Length(min = 4, max = 15)])

    submit = SubmitField('Update')
    def validate_username(self, username):
        '''
        Raises a validation error if a user tries to register using an existing username
        '''
        if username.data != current_user.username:
            user = User.query.filter_by(username = username.data).first()
            if user:
                raise ValidationError('Username Taken')

    def validate_email(self, email):
        '''
        Raises a validation error if a user tries to register using an existing email
        '''
        if email.data != current_user.email:
            user = User.query.filter_by(email = email.data).first()
            if user:
                raise ValidationError('Email Taken')

class RequestResetForm(FlaskForm):
    email = StringField('email', validators = [InputRequired(), Email(message = 'Invalid Email'), Length(max = 50)])
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        '''
        Raises a validation error if a user tries to register using an existing email
        '''
        if email.data != current_user.email:
            user = User.query.filter_by(email = email.data).first()
            if user is None:
                raise ValidationError('There is no accouunt with that email. You must register first.')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators = [DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators = [DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')




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
        # msg = Message(subject = 'Welcome {}'.format(form.username.data), sender = app.config.get("MAIL_USERNAME"), recipients = [str(form.email.data)], body = 'Congratulations you have signed up and your account has been created!')
        # mail.send(msg)

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

@app.route('/learn_more/',methods=['GET', 'POST'])
def learn_more():
    return render_template('learn_more.html')

@app.route('/email_sent/',methods=['GET', 'POST'])
def email_sent():
    return render_template('email_sent.html')

@app.route('/account/',methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()

    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated', 'success')
        return redirect(url_for('account'))

    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email

    return render_template('account.html', title = 'Account', form = form)

@app.route('/model_page/', methods = ['GET','POST'])
@login_required
def model_page():
    return render_template('model_page.html')

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message(subject = 'Password Reset Request',
                  sender = 'noreply@syndicate.com',
                  recipients=[user.email])
    msg.body = f''' To reset your password, visit the following link :
{url_for('reset_token', token = token, _external = True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)


@app.route('/reset_password/',methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email = form.email.data).first()
        flask('An email has been sent with instructions to resset your password', 'info')
        return redirect(url_for('login'))

    return render_template('reset_request.html', title = 'Rest Password', form = form)

@app.route('/reset_password/<token>',methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid / expired token', 'warning')
        return redirect(url_for('reset_request'))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method = 'sha256') # sha256 will generate a hash which is 80 chars long
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated!', 'success')
        # send congrat email for registering
        # msg = Message(subject = 'Welcome {}'.format(form.username.data), sender = app.config.get("MAIL_USERNAME"), recipients = [str(form.email.data)], body = 'Congratulations you have signed up and your account has been created!')
        # mail.send(msg)
        return redirect(url_for('login'))
    return render_template('reset_token.html', title = 'Rest Password', form = form)



@app.route('/predict_model', methods=['GET', 'POST'])
def predict_model():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    map_dict = {1 : 'DT Toronto', 3 : 'North York', 4 : 'Scarborough', 6 : 'Etobicoke'}
    output = map_dict[output]
    return render_template('model_page.html', prediction_text = 'The Crime Occurred in : {}'.format(output))

if __name__ == "__main__":
    if ENV == 'prod':
        app.run()
    else:
        app.run(debug=True)
