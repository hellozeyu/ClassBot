from os import environ
from errors import InvalidUsage
import requests
import boto3


def connect():
    """
    connect to s3
    :return: s3 connection
    """
    return boto3.client('s3',
                        aws_access_key_id=environ.get('S3_KEY_ID'),
                        aws_secret_access_key=environ.get('S3_ACCESS_KEY'))


def get_key(filename, production=False):
    """
    get aws key from filename
    :param filename: filename
    :param production: bool to always get production file
    :return: aws key
    """
    # if environ.get('PIPELINE_ENV') == 'production' or production:
    #     return filename
    # return 'development_files/' + filename
    return filename


def validate_url(url):
    """
    validate s3 url to see if it is working or not
    :param url: generated url
    :return: True if validated
    """
    status_code = requests.get(url).status_code
    if status_code >= 400:
        return False
    return True


def list_files(bucket=None, production=False):
    """
    :param user_id: user id
    :param bucket: bucket name
    :param production: bool to always get production file
    :return: A list of s3 objects
    """
    if bucket is None:
        bucket = environ.get('S3_BUCKET')
    s3_conn = connect()
    response = s3_conn.list_objects_v2(
            Bucket=bucket,
            MaxKeys=100)
    return response.get('Contents', [])


def delete(filename, bucket=None):
    if bucket is None:
        bucket = environ.get('S3_BUCKET')
    s3_conn = connect()
    try:
        response = s3_conn.delete_object(
                Bucket=bucket,
                Key=filename)
    except:
        pass
    # return response.get('Contents', [])


def download(filename, bucket=None, public=False, production=False):
    """
    Generate url for files in private bucket
    :param filename: file name
    :param bucket: bucket name
    :param public: get public url
    :param production: bool to always get production file
    :return: url
    """
    key = get_key(filename, production)
    if bucket is None:
        bucket = environ.get('S3_BUCKET')
    try:
        if public:
            return 'https://%s.s3.amazonaws.com/%s' % (bucket, key)
        s3_conn = connect()
        url = s3_conn.generate_presigned_url('get_object',
                                             Params={'Bucket': bucket, 'Key': key},
                                             ExpiresIn=60)
        return url
    except Exception:
        raise InvalidUsage('Not Found!', status_code=404)


def upload(file, filename, retries=1, bucket=None, public=False, production=False, **kwargs):
    """
    Upload file to S3
    :param file: file object
    :param filename: file name
    :param retries: number of retries
    :param bucket: bucket name
    :param public: to upload a public file
    :param production: bool to always get production file
    :param kwargs: kwargs
    :return: url
    """
    key = get_key(filename, production)
    if bucket is None:
        bucket = environ.get('S3_BUCKET')
    try:
        s3_conn = connect()
        if public:
            kwargs['ACL'] = 'public-read'
            kwargs['CacheControl'] = 'no-cache'
        s3_conn.put_object(Body=file, Bucket=bucket, Key=key, **kwargs)
        url = download(filename=filename, bucket=bucket, public=public)
        if validate_url(url):
            return url
        if retries:
            return upload(file=file, filename=filename, retries=retries - 1,
                          bucket=bucket, public=public, **kwargs)
        raise InvalidUsage('not valid link', status_code=404)
    except Exception:
        if retries:
            return upload(file=file, filename=filename, retries=retries - 1,
                          bucket=bucket, public=public, **kwargs)
        raise InvalidUsage('Your file was not successfully uploaded. Please try again.',
                           status_code=409)