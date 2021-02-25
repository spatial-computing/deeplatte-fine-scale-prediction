import copy
#from shapely.wkt import loads

import geopandas as gpd
from data_models.osm_model import *



NEW_YORK_BOUNDING_BOX = 'POLYGON((-74.25385 40.49709,-74.25385 40.91559,-73.69972 40.91559,-73.69972 40.49709,-74.25385 40.49709))'

LOS_ANGELES_BOUNDING_BOX = 'POLYGON((-118.66638 33.704788,-118.66638 34.33808,-118.160732 34.33808,-118.160732 33.704788,-118.66638 33.704788))'

CHICAGO_BOUNDING_BOX = 'POLYGON((-87.937778 41.644808,-87.937778 42.016399,-87.525004 42.016399,-87.525004 41.644808,-87.937778 41.644808))'

HOUSTON_BOUNDING_BOX = 'POLYGON((-95.801782 29.524622,-95.801782 30.104439,-95.075927 30.104439,-95.075927 29.524622,-95.801782 29.524622))'

PHOENIX_BOUNDING_BOX = 'POLYGON((-112.332164 33.288019,-112.332164 33.920816,-111.927196 33.920816,-111.927196 33.288019,-112.332164 33.288019))'

PHILADELPHIA_BOUNDING_BOX = 'POLYGON((-75.280666 39.873091,-75.280666 40.137851,-74.95675 40.137851,-74.95675 39.873091,-75.280666 39.873091))'

SAN_ANTONIO_BOUNDING_BOX = 'POLYGON((-98.804956 29.228096,-98.804956 29.732424,-98.269759 29.732424,-98.269759 29.228096,-98.804956 29.228096))'

SAN_DIEGO_BOUNDING_BOX = 'POLYGON((-117.281816 32.533696,-117.281816 33.115036,-116.908227 33.115036,-116.908227 32.533696,-117.281816 32.533696))'

DALLAS_BOUNDING_BOX = 'POLYGON((-96.994973 32.619111,-96.994973 33.016802,-96.556132 33.016802,-96.556132 32.619111,-96.994973 32.619111))'

SAN_JOSE_BOUNDING_BOX = 'POLYGON((-122.040062 37.126854,-122.040062 37.469176,-121.589135 37.469176,-121.589135 37.126854,-122.040062 37.126854))'


NEW_YORK_OSM = {
    'buildings_a': NewYorkOsmBuildingA,
    'landuse_a': NewYorkOsmLanduseA,
    'natural': NewYorkOsmNatural,
    'natural_a': NewYorkOsmNaturalA,
    'places': NewYorkOsmPlaces,
    'places_a': NewYorkOsmPlacesA,
    'pois': NewYorkOsmPois,
    'pois_a': NewYorkOsmPoisA,
    'pofw': NewYorkOsmPofw,
    'pofw_a': NewYorkOsmPofwA,
    'railways': NewYorkOsmRailways,
    'roads': NewYorkOsmRoads,
    'traffic': NewYorkOsmTraffic,
    'traffic_a': NewYorkOsmTrafficA,
    'transport': NewYorkOsmTransport,
    'transport_a': NewYorkOsmTransportA,
    'water_a': NewYorkOsmWaterA,
    'waterways': NewYorkOsmWaterway
}

LOS_ANGELES_OSM = {
    'buildings_a': SouthernCaliforniaOsmBuildingA,
    'landuse_a': SouthernCaliforniaOsmLanduseA,
    'natural': SouthernCaliforniaOsmNatural,
    'natural_a': SouthernCaliforniaOsmNaturalA,
    'places': SouthernCaliforniaOsmPlaces,
    'places_a': SouthernCaliforniaOsmPlacesA,
    'pois': SouthernCaliforniaOsmPois,
    'pois_a': SouthernCaliforniaOsmPoisA,
    'pofw': SouthernCaliforniaOsmPofw,
    'pofw_a': SouthernCaliforniaOsmPofwA,
    'railways': SouthernCaliforniaOsmRailways,
    'roads': SouthernCaliforniaOsmRoads,
    'traffic': SouthernCaliforniaOsmTraffic,
    'traffic_a': SouthernCaliforniaOsmTrafficA,
    'transport': SouthernCaliforniaOsmTransport,
    'transport_a': SouthernCaliforniaOsmTransportA,
    'water_a': SouthernCaliforniaOsmWaterA,
    'waterways': SouthernCaliforniaOsmWaterway
}

CHICAGO_OSM = {
    'buildings_a': IllinoisOsmBuildingA,
    'landuse_a': IllinoisOsmLanduseA,
    'natural': IllinoisOsmNatural,
    'natural_a': IllinoisOsmNaturalA,
    'places': IllinoisOsmPlaces,
    'places_a': IllinoisOsmPlacesA,
    'pois': IllinoisOsmPois,
    'pois_a': IllinoisOsmPoisA,
    'pofw': IllinoisOsmPofw,
    'pofw_a': IllinoisOsmPofwA,
    'railways': IllinoisOsmRailways,
    'roads': IllinoisOsmRoads,
    'traffic': IllinoisOsmTraffic,
    'traffic_a': IllinoisOsmTrafficA,
    'transport': IllinoisOsmTransport,
    'transport_a': IllinoisOsmTransportA,
    'water_a': IllinoisOsmWaterA,
    'waterways': IllinoisOsmWaterway
}

HOUSTON_OSM = {
    'buildings_a': TexasOsmBuildingA,
    'landuse_a': TexasOsmLanduseA,
    'natural': TexasOsmNatural,
    'natural_a': TexasOsmNaturalA,
    'places': TexasOsmPlaces,
    'places_a': TexasOsmPlacesA,
    'pois': TexasOsmPois,
    'pois_a': TexasOsmPoisA,
    'pofw': TexasOsmPofw,
    'pofw_a': TexasOsmPofwA,
    'railways': TexasOsmRailways,
    'roads': TexasOsmRoads,
    'traffic': TexasOsmTraffic,
    'traffic_a': TexasOsmTrafficA,
    'transport': TexasOsmTransport,
    'transport_a': TexasOsmTransportA,
    'water_a': TexasOsmWaterA,
    'waterways': TexasOsmWaterway
}

PHOENIX_OSM = {
    'buildings_a': ArizonaOsmBuildingA,
    'landuse_a': ArizonaOsmLanduseA,
    'natural': ArizonaOsmNatural,
    'natural_a': ArizonaOsmNaturalA,
    'places': ArizonaOsmPlaces,
    'places_a': ArizonaOsmPlacesA,
    'pois': ArizonaOsmPois,
    'pois_a': ArizonaOsmPoisA,
    'pofw': ArizonaOsmPofw,
    'pofw_a': ArizonaOsmPofwA,
    'railways': ArizonaOsmRailways,
    'roads': ArizonaOsmRoads,
    'traffic': ArizonaOsmTraffic,
    'traffic_a': ArizonaOsmTrafficA,
    'transport': ArizonaOsmTransport,
    'transport_a': ArizonaOsmTransportA,
    'water_a': ArizonaOsmWaterA,
    'waterways': ArizonaOsmWaterway
}

PHILADELPHIA_OSM = {
    'buildings_a': PennsylvaniaOsmBuildingA,
    'landuse_a': PennsylvaniaOsmLanduseA,
    'natural': PennsylvaniaOsmNatural,
    'natural_a': PennsylvaniaOsmNaturalA,
    'places': PennsylvaniaOsmPlaces,
    'places_a': PennsylvaniaOsmPlacesA,
    'pois': PennsylvaniaOsmPois,
    'pois_a': PennsylvaniaOsmPoisA,
    'pofw': PennsylvaniaOsmPofw,
    'pofw_a': PennsylvaniaOsmPofwA,
    'railways': PennsylvaniaOsmRailways,
    'roads': PennsylvaniaOsmRoads,
    'traffic': PennsylvaniaOsmTraffic,
    'traffic_a': PennsylvaniaOsmTrafficA,
    'transport': PennsylvaniaOsmTransport,
    'transport_a': PennsylvaniaOsmTransportA,
    'water_a': PennsylvaniaOsmWaterA,
    'waterways': PennsylvaniaOsmWaterway
}

SAN_ANTONIO_OSM = {
    'buildings_a':TexasOsmBuildingA,
    'landuse_a':TexasOsmLanduseA,
    'natural':TexasOsmNatural,
    'natural_a':TexasOsmNaturalA,
    'places':TexasOsmPlaces,
    'places_a':TexasOsmPlacesA,
    'pois':TexasOsmPois,
    'pois_a':TexasOsmPoisA,
    'pofw':TexasOsmPofw,
    'pofw_a':TexasOsmPofwA,
    'railways':TexasOsmRailways,
    'roads':TexasOsmRoads,
    'traffic':TexasOsmTraffic,
    'traffic_a': TexasOsmTrafficA,
    'transport': TexasOsmTransport,
    'transport_a': TexasOsmTransportA,
    'water_a': TexasOsmWaterA,
    'waterways': TexasOsmWaterway
}

SAN_DIEGO_OSM = {
    'buildings_a': SouthernCaliforniaOsmBuildingA,
    'landuse_a': SouthernCaliforniaOsmLanduseA,
    'natural': SouthernCaliforniaOsmNatural,
    'natural_a': SouthernCaliforniaOsmNaturalA,
    'places': SouthernCaliforniaOsmPlaces,
    'places_a': SouthernCaliforniaOsmPlacesA,
    'pois': SouthernCaliforniaOsmPois,
    'pois_a': SouthernCaliforniaOsmPoisA,
    'pofw': SouthernCaliforniaOsmPofw,
    'pofw_a': SouthernCaliforniaOsmPofwA,
    'railways': SouthernCaliforniaOsmRailways,
    'roads': SouthernCaliforniaOsmRoads,
    'traffic': SouthernCaliforniaOsmTraffic,
    'traffic_a': SouthernCaliforniaOsmTrafficA,
    'transport': SouthernCaliforniaOsmTransport,
    'transport_a': SouthernCaliforniaOsmTransportA,
    'water_a': SouthernCaliforniaOsmWaterA,
    'waterways': SouthernCaliforniaOsmWaterway
}

DALLAS_OSM = {
    'buildings_a': TexasOsmBuildingA,
    'landuse_a': TexasOsmLanduseA,
    'natural': TexasOsmNatural,
    'natural_a': TexasOsmNaturalA,
    'places': TexasOsmPlaces,
    'places_a': TexasOsmPlacesA,
    'pois': TexasOsmPois,
    'pois_a': TexasOsmPoisA,
    'pofw': TexasOsmPofw,
    'pofw_a': TexasOsmPofwA,
    'railways': TexasOsmRailways,
    'roads': TexasOsmRoads,
    'traffic': TexasOsmTraffic,
    'traffic_a': TexasOsmTrafficA,
    'transport': TexasOsmTransport,
    'transport_a': TexasOsmTransportA,
    'water_a': TexasOsmWaterA,
    'waterways': TexasOsmWaterway
}

SAN_JOSE_OSM = {
    'buildings_a': NorthernCaliforniaOsmBuildingA,
    'landuse_a': NorthernCaliforniaOsmLanduseA,
    'natural': NorthernCaliforniaOsmNatural,
    'natural_a': NorthernCaliforniaOsmNaturalA,
    'places': NorthernCaliforniaOsmPlaces,
    'places_a': NorthernCaliforniaOsmPlacesA,
    'pois': NorthernCaliforniaOsmPois,
    'pois_a': NorthernCaliforniaOsmPoisA,
    'pofw': NorthernCaliforniaOsmPofw,
    'pofw_a': NorthernCaliforniaOsmPofwA,
    'railways': NorthernCaliforniaOsmRailways,
    'roads': NorthernCaliforniaOsmRoads,
    'traffic': NorthernCaliforniaOsmTraffic,
    'traffic_a': NorthernCaliforniaOsmTrafficA,
    'transport': NorthernCaliforniaOsmTransport,
    'transport_a': NorthernCaliforniaOsmTransportA,
    'water_a': NorthernCaliforniaOsmWaterA,
    'waterways': NorthernCaliforniaOsmWaterway
}


NEW_YORK = {
    'OSM': NEW_YORK_OSM,
    'BOUNDING_BOX': NEW_YORK_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': NewYork1000mGridGeoFeature,
}

LOS_ANGELES = {
    'OSM': LOS_ANGELES_OSM,
    'BOUNDING_BOX': LOS_ANGELES_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': LosAngeles1000mGridGeoFeature,
}

CHICAGO = {
    'OSM': CHICAGO_OSM,
    'BOUNDING_BOX': CHICAGO_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': Chicago1000mGridGeoFeature,
}

HOUSTON = {
    'OSM': HOUSTON_OSM,
    'BOUNDING_BOX': HOUSTON_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': Houston1000mGridGeoFeature,
}

PHOENIX = {
    'OSM': PHOENIX_OSM,
    'BOUNDING_BOX': PHOENIX_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': Phoenix1000mGridGeoFeature,
}

PHILADELPHIA = {
    'OSM': PHILADELPHIA_OSM,
    'BOUNDING_BOX': PHILADELPHIA_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': Philadelphia1000mGridGeoFeature,
}

SAN_ANTONIO = {
    'OSM': SAN_ANTONIO_OSM,
    'BOUNDING_BOX': SAN_ANTONIO_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': SanAntonio1000mGridGeoFeature,
}

SAN_DIEGO = {
    'OSM': SAN_DIEGO_OSM,
    'BOUNDING_BOX': SAN_DIEGO_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': SanDiego1000mGridGeoFeature,
}

DALLAS = {
    'OSM': DALLAS_OSM,
    'BOUNDING_BOX': DALLAS_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': Dallas1000mGridGeoFeature,
}

SAN_JOSE = {
    'OSM': SAN_JOSE_OSM,
    'BOUNDING_BOX': SAN_JOSE_BOUNDING_BOX,
    'GRID_OBJ_1000': grid_1000m,
    'GEO_FEATURE_OBJ_1000': SanJose1000mGridGeoFeature,
}


OSM_MODEL = [
    'buildings_a'
    'landuse_a',
    'natural',
    'natural_a',
    'places',
    'places_a',
    'pois',
    'pois_a',
    'pofw',
    'pofw_a',
    'railways',
    'roads',
    'traffic',
    'traffic_a',
    'transport',
    'transport_a',
    'water_a',
    'waterways'
]





