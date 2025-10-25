import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import secrets
import jwt
from datetime import datetime, timedelta

class GoogleAuth:
    def __init__(self):
        self.client_id = os.getenv('GOOGLE_CLIENT_ID')
        self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        self.redirect_uri = os.getenv('GOOGLE_REDIRECT_URI')
        self.scopes = [
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile',
        ]
        self.state = secrets.token_urlsafe(32)

    def get_authorization_url(self):
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri],
                }
            },
            scopes=self.scopes,
        )
        flow.redirect_uri = self.redirect_uri
        authorization_url, _ = flow.authorization_url(
            access_type='offline',
            state=self.state,
            include_granted_scopes='true'
        )
        return authorization_url

    def get_user_info(self, code):
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri],
                }
            },
            scopes=self.scopes,
        )
        flow.redirect_uri = self.redirect_uri
        flow.fetch_token(code=code)
        credentials = flow.credentials

        import google.auth.transport.requests
        import requests

        session = requests.Session()
        session.verify = True

        auth_req = google.auth.transport.requests.Request(session=session)
        id_info = requests.get(
            'https://www.googleapis.com/oauth2/v1/userinfo',
            headers={'Authorization': f'Bearer {credentials.token}'}
        ).json()

        return id_info

def send_verification_email(email):
    """Send verification email to user"""
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_username = os.getenv('SMTP_USERNAME')
    smtp_password = os.getenv('SMTP_PASSWORD')
    
    # Create verification token
    token = create_verification_token(email)
    verification_link = f"{os.getenv('APP_URL')}/verify/{token}"
    
    # Create email message
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = email
    msg['Subject'] = "Verify your Resume Insight account"
    
    body = f"""
    Welcome to Resume Insight!
    
    Please click the link below to verify your email address:
    {verification_link}
    
    This link will expire in 24 hours.
    
    If you did not create an account with Resume Insight, please ignore this email.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Send email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send verification email: {str(e)}")
        return False

def create_verification_token(email):
    """Create a JWT token for email verification"""
    secret = os.getenv('JWT_SECRET_KEY')
    token = jwt.encode({
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, secret, algorithm='HS256')
    return token

def verify_token(token):
    """Verify the email verification token"""
    try:
        secret = os.getenv('JWT_SECRET_KEY')
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        return payload['email']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None