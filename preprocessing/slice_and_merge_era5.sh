#!/bin/bash

LON_MIN=$1
LON_MAX=$2
LAT_MIN=$3
LAT_MAX=$4
INTERVAL=$5
INPUT_PATH_PHASE_1=$6
OUTPUT_PATH_PHASE_1=$7
PREFIX=$8

lon_min_era5=$(echo $LON_MIN-3\*$INTERVAL | bc)
lon_max_era5=$(echo $LON_MAX+3\*$INTERVAL | bc)
lat_min_era5=$(echo $LAT_MIN-3\*$INTERVAL | bc)
lat_max_era5=$(echo $LAT_MAX+3\*$INTERVAL | bc)

echo $lon_min_era5
echo $lon_max_era5
echo $lat_min_era5
echo $lat_max_era5

cd ${INPUT_PATH_PHASE_1}

## slice each file to the desired lon and lat window
for v in 'q' 't' 'u' 'v' 'z' ; do
	files=$(ls "${v}_"*".nc")
	for file in $files ; do
		cdo sellonlatbox,$lon_min_era5,$lon_max_era5,$lat_min_era5,$lat_max_era5 "${INPUT_PATH_PHASE_1}"${file}"" "${OUTPUT_PATH_PHASE_1}${PREFIX}${file}"
	done
done

echo "Done"

cd ${OUTPUT_PATH_PHASE_1}

# merge all the sliced files into a single file for all the time span considered
for v in 'q' 't' 'u' 'v' 'z' ; do
	cdo -O -f nc4 -z zip -L -b F32 mergetime "${PREFIX}${v}_2001.nc" "${PREFIX}${v}_2002.nc" "${PREFIX}${v}_2003.nc" "${PREFIX}${v}_2004.nc" "${PREFIX}${v}_2005.nc" "${PREFIX}${v}_2006.nc" "${PREFIX}${v}_2007.nc" "${PREFIX}${v}_2008.nc" "${PREFIX}${v}_2009.nc" "${PREFIX}${v}_2010.nc" "${PREFIX}${v}_2011.nc" "${PREFIX}${v}_2012.nc" "${PREFIX}${v}_2013.nc" "${PREFIX}${v}_2014.nc" "${PREFIX}${v}_2015.nc" "${PREFIX}${v}_2016.nc" "${PREFIX}${v}.nc"
	# remove temporary files no longer usefule after merging
	files=$(ls ${PREFIX}${v}_*.nc)
	for file in $files ; do
		rm ${file}
	done
done

