#!/bin/bash

LON_MIN=$1
LON_MAX=$2
LAT_MIN=$3
LAT_MAX=$4
INTERVAL=$5
INPUT_PATH_PHASE_1A=$6
OUTPUT_PATH=$7
PREFIX=$8

lon_min_=$(echo $LON_MIN-3\*$INTERVAL | bc)
lon_max_=$(echo $LON_MAX+3\*$INTERVAL | bc)
lat_min_=$(echo $LAT_MIN-3\*$INTERVAL | bc)
lat_max_=$(echo $LAT_MAX+3\*$INTERVAL | bc)

echo $lon_min_
echo $lon_max_
echo $lat_min_
echo $lat_max_

## for each folder ${variable}${level} merge all the files correspionding to the different years
for v in 'hus' 'ta' 'ua' 'va' 'zg' ; do
##for v in 'hus' 'zg'; do
	for l in '1000' '850' '700' '500' '200'; do
		cd "${INPUT_PATH_PHASE_1A}/${v}${l}/year/"
		cdo -O -f nc4 -z zip -L -b F32 mergetime *.nc "${OUTPUT_PATH}${v}${l}.nc"
	done
done

## merge all the precipitation files correspionding to the different years
cd "${INPUT_PATH_PHASE_1A}/pr/year/"
cdo -O -f nc4 -z zip -L -b F32 mergetime *.nc "${OUTPUT_PATH}pr.nc"

cd ${OUTPUT_PATH}

## merge the different levels
for v in 'hus' 'ta' 'ua' 'va' 'zg' ; do
#for v in 'hus' 'zg'; do
	cdo -O -f nc4 -z zip -L -b F32 merge "${v}1000.nc" "${v}850.nc" "${v}700.nc" "${v}500.nc" "${v}200.nc" "${v}.nc"
	# remove temporary files no longer usefule after merging
	rm "${v}1000.nc" "${v}850.nc" "${v}700.nc" "${v}500.nc" "${v}200.nc"
done

## slice each file to the desired lon and lat window
for v in 'hus' 'ta' 'ua' 'va' 'zg' ; do
#for v in 'hus' 'zg'; do
	cdo sellonlatbox,$lon_min_,$lon_max_,$lat_min_,$lat_max_ "${v}.nc" "${PREFIX}${v}.nc"
	rm "${v}.nc"
done

echo "PHASE 1 COMPLETED"



