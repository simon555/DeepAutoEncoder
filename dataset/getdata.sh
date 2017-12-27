fileURL='https://drive.google.com/uc?export=download&id=1Trr5W7a5h5F6t7iULKZMv7Ox4RE6M02I'

ZIPTOOL="unzip"



# GloVe
echo $fileURL
mkdir raw3DVolumes
curl -LO $fileURL
$ZIPTOOL test-file.rar -d raw3DVOLUMES/
rm test-file.zip
