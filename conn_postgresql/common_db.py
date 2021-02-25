from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2
from sqlalchemy.schema import MetaData
from geoalchemy2 import Geometry

pwd = r"m\\tC7;cc"
connection_string = 'postgresql+psycopg2://{usr}:{pwd}@jonsnow.usc.edu/air_quality_prod'\
    .format(usr='eva', pwd='m\\tC7;cc')

engine = create_engine(connection_string, echo=False)

Session = sessionmaker(bind=engine, expire_on_commit=False)

Base = declarative_base()

session = Session()
meta = MetaData()
meta.reflect(bind=engine)

connection = psycopg2.connect(user="eva",
                              password="m\\tC7;cc",
                              host="jonsnow.usc.edu",
                              port="5432",
                              database="air_quality_prod")
cursor = connection.cursor()