import os
import json
from base.settings.base import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, '../db_config/')

db_json = open(os.path.join(CONFIG_DIR, 'database_settings_ops.json')).read()
DATABASES = json.loads(db_json)

CORS_ORIGIN_ALLOW_ALL = True
CORS_ALLOW_CREDENTIALS = True

# MODEL_API_URL = 'http://10.70.189.6:8090/'
MODEL_API_URL = 'http://innerapi.wehago.com/model-engine/'
UI_URL = 'http://10.106.6.61:3000/'
