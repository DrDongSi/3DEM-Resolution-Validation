#!/bin/bash

COMMON_TMP_DATA_ZIP="./common/data/data.zip"
if [ -f COMMON_TMP_DATA_ZIP ]; then
	rm -f COMMON_TMP_DATA_ZIP
fi
cat ./common/data/*.zip* > COMMON_TMP_DATA_ZIP
unzip -o COMMON_TMP_DATA_ZIP -d ./common/data/
rm -f COMMON_TMP_DATA_ZIP

MONORES_TMP_DATA_ZIP="./monores/data/data.zip"
if [ -f MONORES_TMP_DATA_ZIP ]; then
	rm -f MONORES_TMP_DATA_ZIP
fi
cat ./monores/data/*.zip* > MONORES_TMP_DATA_ZIP
unzip -o MONORES_TMP_DATA_ZIP -d ./monores/data/
rm -f MONORES_TMP_DATA_ZIP

RESMAP_TMP_DATA_ZIP="./resmap/data/data.zip"
if [ -f RESMAP_TMP_DATA_ZIP ]; then
	rm -f RESMAP_TMP_DATA_ZIP
fi
cat ./resmap/data/*.zip* > RESMAP_TMP_DATA_ZIP
unzip -o RESMAP_TMP_DATA_ZIP -d ./resmap/data/
rm -f RESMAP_TMP_DATA_ZIP