#!/bin/bash

explist=(
c01_quantile_mapping_disagregated_out.nc
p07a_n01_l01_i01_b01_r01_012_out.nc
p07a_n01_l01_i02_b01_r01_012_out.nc
p07a_n01_l01_i03_b01_r01_012_out.nc
p07a_n01_l02_i01_b01_r01_012_out.nc
p07a_n01_l02_i02_b01_r01_012_out.nc
p07a_n01_l02_i03_b01_r01_012_out.nc
p07a_n02_l01_i01_b01_r01_012_out.nc
p07a_n02_l01_i02_b01_r01_012_out.nc
p07a_n02_l01_i03_b01_r01_012_out.nc
p07a_n02_l02_i01_b01_r01_012_out.nc
p07a_n02_l02_i02_b01_r01_012_out.nc
p07a_n02_l02_i03_b01_r01_012_out.nc
p07a_n03_l01_i01_b01_r01_012_out.nc
p07a_n03_l01_i02_b01_r01_012_out.nc
p07a_n03_l01_i03_b01_r01_012_out.nc
p07a_n03_l02_i01_b01_r01_012_out.nc
p07a_n03_l02_i02_b01_r01_012_out.nc
p07a_n03_l02_i03_b01_r01_012_out.nc
p07a_s01_l01_i01_b01_r01_012_out.nc
p07a_s01_l01_i02_b01_r01_012_out.nc
p07a_s01_l01_i03_b01_r01_012_out.nc
p07a_s01_l02_i01_b01_r01_012_out.nc
p07a_s01_l02_i02_b01_r01_012_out.nc
p07a_s01_l02_i03_b01_r01_012_out.nc
p07a_s02_l01_i01_b01_r01_012_out.nc
p07a_s02_l01_i02_b01_r01_012_out.nc
p07a_s02_l01_i03_b01_r01_012_out.nc
p07a_s02_l02_i01_b01_r01_012_out.nc
p07a_s02_l02_i02_b01_r01_012_out.nc
p07a_s02_l02_i03_b01_r01_012_out.nc
p07a_s03_l01_i01_b01_r01_012_out.nc
p07a_s03_l01_i02_b01_r01_012_out.nc
p07a_s03_l01_i03_b01_r01_012_out.nc
p07a_s03_l02_i01_b01_r01_012_out.nc
p07a_s03_l02_i02_b01_r01_012_out.nc
p07a_s03_l02_i03_b01_r01_012_out.nc
p07a_u01_l01_i01_b01_r01_012_out.nc
p07a_u01_l01_i02_b01_r01_012_out.nc
p07a_u01_l01_i03_b01_r01_012_out.nc
p07a_u01_l02_i01_b01_r01_012_out.nc
p07a_u01_l02_i02_b01_r01_012_out.nc
p07a_u01_l02_i03_b01_r01_012_out.nc
p07a_u02_l01_i01_b01_r01_012_out.nc
p07a_u02_l01_i02_b01_r01_012_out.nc
p07a_u02_l01_i03_b01_r01_012_out.nc
p07a_u02_l02_i01_b01_r01_012_out.nc
p07a_u02_l02_i02_b01_r01_012_out.nc
p07a_u02_l02_i03_b01_r01_012_out.nc
ref_imda_012_data.nc
)

################################################## P07A_RMEAN_v1.nc ################################################

#------------------------------------------------------------------------------------------------------
varname=RMEANMON
SOURCEPATH=/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA
DESTINPATH=/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA/CLIMATE_INDICES_DATA
#------------------------------------------------------------------------------------------------------
echo Processing ... ${varname}

DIR=${DESTINPATH}/${varname}

if [ "$1" = "c" ]; then
    rm -rfv $DIR
fi

if [ "$2" = "d" ]; then
    rm -v ${DESTINPATH}/${OUTFILE}
fi

mkdir -p "${DIR}"

for exp_id in ${explist[@]}; do
        filepath=$(find ${SOURCEPATH} -maxdepth 1 -name "${exp_id}" -type f | sed 's/\.\///')
        filename=$(basename "$filepath")
        echo "${filename}"
        cdo -v -L \
	-setname,"prec" \
        -monmean "${SOURCEPATH}/${filename}" \
        "${DIR}/${filename}"
        echo ''
done



################################################# P07A_RX1DAY_v1.nc ################################################

#------------------------------------------------------------------------------------------------------
varname=RX1DAYMON
SOURCEPATH=/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA
DESTINPATH=/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA/CLIMATE_INDICES_DATA
#------------------------------------------------------------------------------------------------------
echo Processing ... ${varname}

DIR=${DESTINPATH}/${varname}

if [ "$1" = "c" ]; then
    rm -rfv $DIR
fi

if [ "$2" = "d" ]; then
    rm -v ${DESTINPATH}/${OUTFILE}
fi

mkdir -p "${DIR}"


for exp_id in ${explist[@]}; do
        filepath=$(find ${SOURCEPATH} -maxdepth 1 -name "${exp_id}" -type f | sed 's/\.\///')
        filename=$(basename "$filepath")
        echo "${filename}"
        cdo -v -L \
	-setname,"prec" \
        -etccdi_rx1day,freq=month "${SOURCEPATH}/${filename}" \
        "${DIR}/${filename}"
        echo ''
done

################################################## P07A_R{PCTL}_v1.nc ################################################

for pctl in 99 95; do # 90 80 75 50 25 2 1 98

#------------------------------------------------------------------------------------------------------
varname=R${pctl}MON
SOURCEPATH=/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA
DESTINPATH=/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA/CLIMATE_INDICES_DATA
#------------------------------------------------------------------------------------------------------
echo Processing ... ${varname}

DIR=${DESTINPATH}/${varname}

if [ "$1" = "c" ]; then
    rm -rfv $DIR
fi

if [ "$2" = "d" ]; then
    rm -v ${DESTINPATH}/${OUTFILE}
fi

mkdir -p "${DIR}"


for exp_id in ${explist[@]}; do
        filepath=$(find ${SOURCEPATH} -maxdepth 1 -name "${exp_id}" -type f | sed 's/\.\///')
        filename=$(basename "$filepath")
        echo "${filename}"
        cdo -v -L \
	-setname,"prec" \
        -monpctl,${pctl} "${SOURCEPATH}/${filename}" -monmin "${SOURCEPATH}/${filename}" -monmax "${SOURCEPATH}/${filename}" \
        "${DIR}/${filename}"
        echo ''
done
done



