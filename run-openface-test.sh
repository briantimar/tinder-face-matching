#/bin/bash

docker run -it -v $(pwd):/tmp -w /tmp bamos/openface python -i openface-test.py test_images/* --verbose