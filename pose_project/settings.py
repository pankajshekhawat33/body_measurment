"""
Django settings for pose_project project.
"""

from pathlib import Path
import os
import dj_database_url


# ======================================================
# BASE DIRECTORY
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ======================================================
# SECURITY
# ======================================================

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-change-this-key"
)

DEBUG = os.environ.get("DEBUG", "False") == "True"

ALLOWED_HOSTS = ["*"]


# ======================================================
# APPLICATIONS
# ======================================================

INSTALLED_APPS = [
    # django apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # third party
    'rest_framework',
    'corsheaders',

    # local apps
    'pose',
]


# ======================================================
# MIDDLEWARE
# ======================================================

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',

    # whitenoise
    'whitenoise.middleware.WhiteNoiseMiddleware',

    # cors
    'corsheaders.middleware.CorsMiddleware',

    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',

    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',

    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# ======================================================
# URLS
# ======================================================

ROOT_URLCONF = 'pose_project.urls'


# ======================================================
# TEMPLATES
# ======================================================

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',

        'DIRS': [BASE_DIR / "templates"],

        'APP_DIRS': True,

        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]


# ======================================================
# WSGI
# ======================================================

WSGI_APPLICATION = 'pose_project.wsgi.application'


# ======================================================
# DATABASE
# ======================================================

DATABASES = {
    'default': dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600
    )
}


# ======================================================
# PASSWORD VALIDATION
# ======================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# ======================================================
# INTERNATIONALIZATION
# ======================================================

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# ======================================================
# STATIC FILES
# ======================================================

STATIC_URL = '/static/'

STATIC_ROOT = BASE_DIR / 'staticfiles'

STATICFILES_STORAGE = (
    'whitenoise.storage.CompressedManifestStaticFilesStorage'
)


# ======================================================
# MEDIA FILES
# ======================================================

MEDIA_URL = '/media/'

MEDIA_ROOT = BASE_DIR / 'media'


# ======================================================
# DEFAULT PRIMARY KEY
# ======================================================

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ======================================================
# CORS
# ======================================================

CORS_ALLOW_ALL_ORIGINS = True

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_HEADERS = [
    'content-type',
    'authorization',
]


# ======================================================
# REST FRAMEWORK
# ======================================================

REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ]
}


# ======================================================
# AI SETTINGS
# ======================================================

BODY_AI_DATASET_PATH = os.path.join(
    BASE_DIR,
    "pose",
    "dataset.csv"
)

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "pose",
    "models",
    "yolov8n.pt"
)


# ======================================================
# FILE UPLOAD LIMITS
# ======================================================

DATA_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100MB

FILE_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100MB


# ======================================================
# LOGGING
# ======================================================

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },

    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}