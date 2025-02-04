######################################### pbias mbias calculation ########################################
#----------------------------------------------------------------------------------------------------

SOURCEPATH="/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/GENDATA"
DESTINPATH="/home/midhunm/AI4KLIM/EXPMNTS/P07A_DeepDown_Comparison/ANALYSE/PLTDATA"
#----------------------------------------------------------------------------------------------------

refer_list=(imda)
model_list=(
c01 
n01_l01_i01
n01_l01_i02
n01_l01_i03
n01_l02_i01
n01_l02_i02
n01_l02_i03
n02_l01_i01
n02_l01_i02
n02_l01_i03
n02_l02_i01
n02_l02_i02
n02_l02_i03
n03_l01_i01
n03_l01_i02
n03_l01_i03
n03_l02_i01
n03_l02_i02
n03_l02_i03
s01_l01_i01
s01_l01_i02
s01_l01_i03
s01_l02_i01
s01_l02_i02
s01_l02_i03
s02_l01_i01
s02_l01_i02
s02_l01_i03
s02_l02_i01
s02_l02_i02
s02_l02_i03
s03_l01_i01
s03_l01_i02
s03_l01_i03
s03_l02_i01
s03_l02_i02
s03_l02_i03
u01_l01_i01
u01_l01_i02
u01_l01_i03
u01_l02_i01
u01_l02_i02
u01_l02_i03
u02_l01_i01
u02_l01_i02
u02_l01_i03
u02_l02_i01
u02_l02_i02
u02_l02_i03
)


for refer in ${refer_list[@]}; do

  for pctl in 90 95 99; do

    DIR="TEMP_${refer}"
    mkdir -p "${DESTINPATH}/${DIR}"

    for model in ${model_list[@]}; do
	
        modelpath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${model}*.nc" -type f | sed 's/\.\///')
	modelname=$(basename "$modelpath")
	
	referpath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${refer}*.nc" -type f | sed 's/\.\///')
	refername=$(basename "$referpath")
        
	echo "Calculating R${pctl}BIAS of ${modelname} w.r.t. ${refername} ..."
	cdo -s -L \
	-setname,${model^^}_R${pctl}BIAS \
	-sub \
	-timpctl,${pctl} ${SOURCEPATH}/${modelname} -timmin ${SOURCEPATH}/${modelname} -timmax ${SOURCEPATH}/${modelname} \
	-timpctl,${pctl} ${SOURCEPATH}/${refername} -timmin ${SOURCEPATH}/${refername} -timmax ${SOURCEPATH}/${refername} \
	${DESTINPATH}/${DIR}/TEMP2DEL_${refer}_${model}_R${pctl}BIAS.nc
    done
  cdo -v -L merge ${DESTINPATH}/${DIR}/TEMP2DEL_*.nc ${DESTINPATH}/EVAL_METRICS_SPATIAL_R${pctl}BIAS.nc
  done
done

#for refer in ${refer_list[@]}; do

#    DIR="MBIAS_PBIAS_WRT_${refer}"
#    mkdir -p "${DESTINPATH}/${DIR}"

#    for model in ${model_list[@]}; do
#	
#        modelpath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${model}*.nc" -type f | sed 's/\.\///')
#	modelname=$(basename "$modelpath")
#	
#	referpath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${refer}*.nc" -type f | sed 's/\.\///')
#	refername=$(basename "$referpath")
#        
#	echo "Calculating PBIAS of ${modelname} w.r.t. ${refername} ..."
#	cdo -s -L \
#	-setname,${model^^}_PBIAS \
#	-setrtoc,-10000000,-200,-200 \
#	-setrtoc,200,10000000,-200 \
#	-mulc,100 \
#	-div \
#	-timsum \
#	-sub ${SOURCEPATH}/${modelname} ${SOURCEPATH}/${refername} \
#	-timsum ${SOURCEPATH}/${refername} \
#	${DESTINPATH}/${DIR}/TEMP2DEL_${refer}_${model}_PBIAS.nc
#	
#	echo "Calculating MBIAS of ${modelname} w.r.t. ${refername} ..."
#	cdo -s -L \
#	-setname,${model}_MBIAS \
#	-timmean \
#	-sub ${SOURCEPATH}/${modelname} ${SOURCEPATH}/${refername} \
#	${DESTINPATH}/${DIR}/TEMP2DEL_${refer}_${model}_MBIAS.nc
#    done

#cdo -v -L merge ${DESTINPATH}/${DIR}/TEMP2DEL_*.nc ${DESTINPATH}/MBIAS_PBIAS_WRT_${refer}.nc
#done











