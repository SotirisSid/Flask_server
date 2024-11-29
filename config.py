class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///../instance/keystroke_dynamics.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'secret'
    JWT_SECRET_KEY = 'secret'
    