fileURL='https://drive.google.com/uc?export=download&id=1Trr5W7a5h5F6t7iULKZMv7Ox4RE6M02I'




# GloVe
echo $fileURL
curl -LO $fileURL
unrar e data-mvct.rar raw3DVolumes/
rm data-mvct.rar
