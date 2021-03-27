from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2
from sqlalchemy.schema import MetaData
import os

load_dotenv('.env')

DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

connection_string = 'postgresql+psycopg2://{usr}:{pwd}@jonsnow.usc.edu/air_quality_prod'\
    .format(usr=DB_USERNAME, pwd=DB_PASSWORD)

engine = create_engine(connection_string, echo=False)

Session = sessionmaker(bind=engine, expire_on_commit=False)

Base = declarative_base()

session = Session()
meta = MetaData()
meta.reflect(bind=engine)

connection = psycopg2.connect(user=DB_USERNAME,
                              password=DB_PASSWORD,
                              host=DB_HOST,
                              port=DB_PORT,
                              database=DB_NAME)
cursor = connection.cursor()