from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2
from sqlalchemy.schema import MetaData
from geoalchemy2 import Geometry
import os

pwd = os.environ['PGPWD']
usr = os.environ['PGUSR']
host = os.environ['PGHOST']
port = os.environ['PGPORT']
db = os.environ['PGDB']


connection_string = 'postgresql+psycopg2://{usr}:{pwd}@jonsnow.usc.edu/air_quality_prod'\
    .format(usr=usr, pwd=pwd)

engine = create_engine(connection_string, echo=False)

Session = sessionmaker(bind=engine, expire_on_commit=False)

Base = declarative_base()

session = Session()
meta = MetaData()
meta.reflect(bind=engine)

connection = psycopg2.connect(user=usr,
                              password=pwd,
                              host=host,
                              port=port,
                              database=db)
cursor = connection.cursor()