# download_wts.sh

# Get pre-trained model wts (1.24G) from gdrive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1D2uYQmrZaZPngDi1U8aSPoXdzuAnEwhb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1D2uYQmrZaZPngDi1U8aSPoXdzuAnEwhb" -O wts.tar.gz && rm -rf /tmp/cookies.txt

# Untar file
echo "Untarring wts.tar.gz"
tar -xf wts.tar.gz
