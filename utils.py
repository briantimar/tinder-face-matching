import requests
import warnings
import io
import PIL

def get_photo_url(photo, size=None):
    """Returns url string corresponding to the json photo object.
        size: if not None, size of processed image to return"""
    #preprocessed file sizes provided by Tinder
    # maps file sizes to indices in preproc photos list
    #note: these are, apparently, only approximate. I don't know if 
    # the range of sizes is standard.
    SIZES = {640: 0, 320: 1, 172: 2, 84: 3}
    # extension = photo['extension']
    if size is None:
        return photo['url']
    if size not in SIZES.keys():
        raise ValueError("Not a valid preprocessed image size: %d" % size)
    preproc = photo['processedFiles'][SIZES[size]]
    # assert preproc['height'] == size
    return preproc['url']

def get_all_photo_urls(person, size=None):
    """ Return list of all photos urls in json object person.
        size: if not None, linear size of each image to return"""
    photos = person['photos']
    return list(map(lambda p: get_photo_url(p, size=size), photos))

def save_image_from_url(image_url, fname):
    """ Save image from the given url to local path fname."""
    extension = image_url.split('.')[-1]
    dat = requests.get(image_url)
    if dat.status_code != 200:
        warnings.warn("Image retrieval from %s failed"%image_url)
    with open(fname + "." + extension, 'wb') as f:
        f.write(dat.content)
    return dat.status_code

def PIL_from_url(image_url):
    """ Return PIL Image from the given url."""
    r = requests.get(image_url)
    if not r.ok:
        warnings.warn("Image retrieval from %s failed"%image_url)
        return None
    return PIL.Image.open(io.BytesIO(r.content))