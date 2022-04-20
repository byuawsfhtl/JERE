import os
import os.path
import sys

if not os.path.exists('BMDRecordGeneration'):
    print('Downloading Record Generator...')
    os.system("""wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bY-iaCn_CTaZE2-wp7_tJWPqocHLTC7y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bY-iaCn_CTaZE2-wp7_tJWPqocHLTC7y" -O BMDRecordGenerator.zip && rm -rf /tmp/cookies.txt""")
    os.system("unzip BMDRecordGenerator && rm BMDRecordGenerator.zip")