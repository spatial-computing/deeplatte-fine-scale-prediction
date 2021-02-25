from conn_postgresql.common_db import engine, session
from conn_postgresql.common_db import connection
import time
import sys
import pandas as pd


def create_pandas_table(sql_query, database=connection):
    table = pd.read_sql_query(sql_query, database)
    return table

def drop_create_table(table_obj):
    try:
        table_obj.__table__.drop(bind=engine, checkfirst=True)
        table_obj.__table__.create(bind=engine, checkfirst=True)
        return
    except Exception as e:
        print(e)
        sys.exit(-1)

def create_table(table_obj):
    try:
        # table_obj.__table__.drop(bind=engine, checkfirst=True)
        table_obj.__table__.create(bind=engine, checkfirst=True)
        return

    except Exception as e:
        print(e)
        #sys.exit(-1)


def insert_locations(data, location_table_obj):
    data_obj = []
    for item in data:
        obj = location_table_obj(
            sensor_id=item.get('sensor_id'),
            parent_id=item.get('parent_id'),
            channel=item.get('channel'),
            label=item.get('label'),
            device_location_type=item.get('device_location_type'),
            thingspeak_primary_id=item.get('thingspeak_primary_id'),
            thingspeak_primary_id_read_key=item.get('thingspeak_primary_id_read_key'),
            thingspeak_second_id=item.get('thingspeak_second_id'),
            thingspeak_second_id_read_key=item.get('thingspeak_second_id_read_key'),
            lon=item.get('lon'),
            lat=item.get('lat'),
            location='SRID=4326;POINT({} {})'.format(item.get('lon'), item.get('lat')))
        data_obj.append(obj)
    print("created obj")
    session.bulk_save_objects(data_obj)
    session.commit()

def insert_ppa_data(data, data_table_obj):
    data_obj = []
    t0 = time.time()
    for item in data:
        if len(item) == 0 or item.get('sensor_id') is None:
            continue
        obj = data_table_obj(
            sensor_id=item.get('sensor_id'),
            channel=item.get('channel'),
            timestamp=item.get('timestamp'),
            pm1_atm=item.get('pm1_atm'),
            pm2_5_atm=item.get('pm2_5_atm'),
            pm10_atm=item.get('pm10_atm'),
            pm1_cf_1=item.get('pm1_cf_1'),
            pm2_5_cf_1=item.get('pm2_5_cf_1'),
            pm10_cf_1=item.get('pm10_cf_1'),
            p_0_3um_cnt=item.get('p_0_3um_cnt'),
            p_0_5um_cnt=item.get('p_0_5um_cnt'),
            p_1_0um_cnt=item.get('p_1_0um_cnt'),
            p_2_5um_cnt=item.get('p_2_5um_cnt'),
            p_5um_cnt=item.get('p_5um_cnt'),
            p_10um_cnt=item.get('p_10um_cnt'),
            rssi=item.get('rssi'),
            temperature=item.get('temperature'),
            humidity=item.get('humidity'))
        data_obj.append(obj)
    print("created obj")
    session.bulk_save_objects(data_obj)
    # session.add_all(data_obj)
    session.commit()
    print("time elapsed: ", time.time() - t0, "secs")

def insert_darksky_data(data, darksky_obj):

    new_data = []

    for item in data:
        obj = darksky_obj(gid=item['gid'],
                          timestamp=item['timestamp'],
                          summary=item.get('summary'),
                          icon=item.get('icon'),
                          precip_intensity=item.get('precipIntensity'),
                          precip_probability=item.get('precipProbability'),
                          temperature=item.get('temperature'),
                          apparent_temperature=item.get('apparentTemperature'),
                          dew_point=item.get('dewPoint'),
                          humidity=item.get('humidity'),
                          pressure=item.get('pressure'),
                          wind_speed=item.get('windSpeed'),
                          wind_bearing=item.get('windBearing'),
                          cloud_cover=item.get('cloudCover'),
                          uv_index=item.get('uvIndex'),
                          visibility=item.get('visibility'),
                          ozone=item.get('ozone'))
        new_data.append(obj)

    session.add_all(new_data)
    session.commit()


def insert_climacell_data(data, darksky_obj):

    new_data = []

    for item in data:
        obj = darksky_obj(gid=item['gid'],
                          timestamp=item['observation_time'],
                          summary=item.get('weather_code'),
                          #icon=item.get('icon'),
                          precip_intensity=item.get('precipitation'),
                          #precip_probability=item.get('precipProbability'),
                          temperature=item.get('temp'),
                          #apparent_temperature=item.get('apparentTemperature'),
                          dew_point=item.get('dewpoint'),
                          humidity=item.get('humidity'),
                          pressure=item.get('baro_pressure'),
                          wind_speed=item.get('wind_speed'),
                          wind_bearing=item.get('wind_direction'),
                          cloud_cover=item.get('cloud_cover'),
                          #uv_index=item.get('uvIndex'),
                          visibility=item.get('visibility'))
                          #ozone=item.get('ozone'))
        new_data.append(obj)

    session.add_all(new_data)
    session.commit()