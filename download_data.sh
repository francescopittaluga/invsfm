# download_data.sh

# Get data files (11G) from gdrive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1StpUiEauckZcxHZeBzoq6L2K7pcB9v3E' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1StpUiEauckZcxHZeBzoq6L2K7pcB9v3E" -O data.tar.gz && rm -rf /tmp/cookies.txt

# Untar file
echo "Untarring data.tar.gz"
tar -xf data.tar.gz
