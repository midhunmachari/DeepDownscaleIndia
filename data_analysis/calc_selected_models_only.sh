#!/bin/bash

SOURCEPATH="/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA"
DESTINPATH="/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA"

explist=(
ref
c01
n01_l01_i03
n02_l01_i03
n03_l01_i03
s01_l01_i03
s02_l01_i03
s03_l01_i03
u01_l01_i03
u02_l01_i03
)

################################################ P07A_ANNUAL_CYCLE_v1.nc ################################################


#----------------------------------------------------------------------------------------------------
varname=SELMODELS
OUTFILE=P07A_${varname}_v1.nc
#----------------------------------------------------------------------------------------------------

echo Processing .. "${varname}"

DIR=TEMP

if [ "$1" = "c" ]; then
    rm -rfv $DIR
fi

if [ "$2" = "d" ]; then
    rm -v ${DESTINPATH}/${OUTFILE}
fi

mkdir -p "${DIR}"


for exp_id in ${explist[@]}; do
        filepath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${exp_id}*.nc" -type f | sed 's/\.\///')
        filename=$(basename "$filepath")
        echo "${filename}"
        cdo -v -L \
        -setname,${exp_id^^} \
	"${SOURCEPATH}/${filename}" \
        "${PWD}/${DIR}/TEMP_${exp_id^^}_${varname}.nc"
        echo ''
done

cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
