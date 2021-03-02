import sys

# sys.path.append('/home/eva/jonsnow_air_quality')
from conn_postgresql.common_db import Base, session, engine, meta

from sqlalchemy import func
import pandas as pd
from sqlalchemy import Table

from utils import LOS_ANGELES_OSM


def get_gridobj(tablename, schema='geographic_data'):
    return Table(tablename, meta, autoload=True, autoload_with=engine, schema=schema)


def construct_geo_vector(**kwargs):
    geo_feature_obj = kwargs['GEO_FEATURE_OBJ']
    coord_obj = kwargs['COORD_OBJ']
    geo_vector_obj = kwargs['GEO_VECTOR_OBJ']
    geo_name_obj = kwargs['GEO_NAME_OBJ']

    locations = sorted([i[0] for i in session.query(coord_obj.gid).all()])
    geo_name_df = pd.read_sql(session.query(geo_name_obj.name).statement, session.bind)

    try:
        for loc in locations:

            geo_data_sql = session.query(geo_feature_obj.value, func.concat(
                geo_feature_obj.geo_feature, '_', geo_feature_obj.feature_type).label('name')) \
                .filter(geo_feature_obj.gid == loc).statement

            geo_data_df = pd.read_sql(geo_data_sql, session.bind)
            geo_data = geo_name_df.merge(geo_data_df, on='name', how='left')
            geo_data = geo_data['value'].fillna(0.0)

            coord = session.query(coord_obj.lon, coord_obj.lat).filter(coord_obj.gid == loc).first()
            obj_result = geo_vector_obj(gid=loc, data=list(geo_data) + list(coord))

            session.add(obj_result)
            session.commit()

            if loc % 1000 == 0:
                print('Geo Vector {} has finished.'.format(len(list(geo_data) + list(coord))))

        # adding lon, lat into geo feature names
        obj_results = [geo_name_obj(name='lon', geo_feature='location', feature_type='lon'),
                       geo_name_obj(name='lat', geo_feature='location', feature_type='lat')]
        # session.add_all(obj_results)
        # session.commit()

        return

    except Exception as e:
        print(e)
        exit(-1)


def construct_geo_name(geo_feature_obj, geo_name_obj):
    try:
        #  filter geographic data by features and feature types

        geo_data = session.query(geo_feature_obj) \
            .filter(geo_feature_obj.c.geo_feature.in_(LOS_ANGELES_OSM)) \
            .filter(~geo_feature_obj.c.feature_type.in_(['unknown', 'unclassified'])).subquery()

        geo_name = session.query(func.concat(geo_data.c.geo_feature, '_', geo_data.c.feature_type).label('name'),
                                 geo_data.c.geo_feature, geo_data.c.feature_type).distinct().order_by('name').all()

        obj_results = [geo_name_obj(name=item[0], geo_feature=item[1], feature_type=item[2]) for item in geo_name]
        # session.add_all(obj_results)
        # session.commit()

        print('Generated {} Geo Names.'.format(len(geo_name)))
        return

    except Exception as e:
        print(e)
        exit(-1)


def main(**kwargs):
    """ !!! Be careful, create table would overwrite the original table """
    # create_table(kwargs['GEO_NAME_OBJ'])
    construct_geo_name(kwargs['GEO_FEATURE_OBJ'], kwargs['GEO_NAME_OBJ'])

    # create_table(kwargs['GEO_VECTOR_OBJ'])
    construct_geo_vector(**kwargs)


if __name__ == '__main__':
    coordobj = get_gridobj(f'grid_{1000}m')
    GEO_FEATURE_OBJ = get_gridobj(f'{"los_angeles"}_grid_{1000}m_geo_feature', "geographic_data")
    GEO_VECTOR_OBJ = get_gridobj(f'{"los_angeles"}_{1000}m_grid_geo_vector', "preprocess")
    GEO_NAME_OBJ = get_gridobj(f'{"los_angeles"}_{1000}m_grid_geo_name', "preprocess")

    configs = [

        {
            'COORD_OBJ': coordobj,
            'GEO_FEATURE_OBJ': GEO_FEATURE_OBJ,
            'GEO_VECTOR_OBJ': GEO_VECTOR_OBJ,
            'GEO_NAME_OBJ': GEO_NAME_OBJ
        }
    ]

    for conf in configs:
        main(**conf)
