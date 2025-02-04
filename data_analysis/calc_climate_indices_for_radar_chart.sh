#!/bin/bash

SOURCEPATH="/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA"
PLDATAPATH="/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA/CLIMATE_INDICES_DATA_RADAR/CLIMATE_INDICES" # Edit here

suffix="L01.I01"

explist=(
ref
c01

n01_l01_i01
n02_l01_i01
n03_l01_i01
s01_l01_i01
s02_l01_i01
s03_l01_i01
u01_l01_i01
u02_l01_i01

# n01_l01_i02
# n02_l01_i02
# n03_l01_i02
# s01_l01_i02
# s02_l01_i02
# s03_l01_i02
# u01_l01_i02
# u02_l01_i02

# n01_l01_i03
# n02_l01_i03
# n03_l01_i03
# s01_l01_i03
# s02_l01_i03
# s03_l01_i03
# u01_l01_i03
# u02_l01_i03

# n01_l02_i01
# n02_l02_i01
# n03_l02_i01
# s01_l02_i01
# s02_l02_i01
# s03_l02_i01
# u01_l02_i01
# u02_l02_i01

# n01_l02_i02
# n02_l02_i02
# n03_l02_i02
# s01_l02_i02
# s02_l02_i02
# s03_l02_i02
# u01_l02_i02
# u02_l02_i02

# n01_l02_i03
# n02_l02_i03
# n03_l02_i03
# s01_l02_i03
# s02_l02_i03
# s03_l02_i03
# u01_l02_i03
# u02_l02_i03

)

DESTINPATH="${PLDATAPATH}.${suffix}"
mkdir -p ${DESTINPATH}

DIR=TEMP_${suffix}

######################################### P07A_CLIMATE_INDICES_R{99/95/90/75/50}MON_v1.nc ########################################

for pctl in 99 98 95 90; do
echo Processing .. R${pctl}MON
#----------------------------------------------------------------------------------------------------
varname=R${pctl}MON
OUTFILE=P07A_CLIMATE_INDICES_${varname}_v1.nc
#----------------------------------------------------------------------------------------------------

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
        -setname,${exp_id^^}_${varname} \
        -monpctl,${pctl} "${SOURCEPATH}/${filename}" -monmin "${SOURCEPATH}/${filename}" -monmax "${SOURCEPATH}/${filename}" \
        "${PWD}/${DIR}/TEMP_${exp_id^^}_${varname}.nc"
        echo ''
done

cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
rm -rv ${DIR}
done

################################################ P07A_CLIMATE_INDICES_RX1DAYMON_v1.nc ################################################

echo Processing .. "RX1DAYMON"
#----------------------------------------------------------------------------------------------------
varname=RX1DAYMON
OUTFILE=P07A_CLIMATE_INDICES_${varname}_v1.nc
#----------------------------------------------------------------------------------------------------

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
        -setname,${exp_id^^}_${varname} \
        -etccdi_rx1day,freq=month "${SOURCEPATH}/${filename}" \
        "${PWD}/${DIR}/TEMP_${exp_id^^}_${varname}.nc"
        echo ''
done

cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
rm -rv ${DIR}

################################################ P07A_CLIMATE_INDICES_RMONMEAN_v1.nc ################################################

echo Processing .. "RMONMEAN"
#----------------------------------------------------------------------------------------------------
varname=RMONMEAN
OUTFILE=P07A_CLIMATE_INDICES_${varname}_v1.nc
#----------------------------------------------------------------------------------------------------

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
       -setname,${exp_id^^}_${varname} \
       -monmean "${SOURCEPATH}/${filename}" \
       "${PWD}/${DIR}/TEMP_${exp_id^^}_${varname}.nc"
       echo ''
done

cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
rm -rv ${DIR}

###############################################################################################################################

echo "Job exit at $(date '+%Y-%m-%d %H:%M:%S')"