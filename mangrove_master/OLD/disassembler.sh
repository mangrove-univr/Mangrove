
echo $PWD/CMakeFiles/Mangrove.dir/Mangrove.sm_35.cubin

if [ -f $PWD/CMakeFiles/Mangrove.dir/Mangrove.sm_35.cubin ]; then
	nvdisasm -c -sf -g $PWD/CMakeFiles/Mangrove.dir/Mangrove.sm_35.cubin > Mangrove.sass
	echo -e "\nGenerationg Mangrove.sass\n"
else
	echo -e "\Mangrove.sm_35.cubin not found\n"
fi
