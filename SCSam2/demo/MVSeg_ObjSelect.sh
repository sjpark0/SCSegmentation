NAME=Blocks1
python sam2_demoVideo_maskInput.py $NAME
python sam2_demoVideo_maskSingleInput.py $NAME
python sam2_demoVideoNew_maskInput.py $NAME
python sam2_demoVideoNew_maskSingleInput.py $NAME
python ComputeIOU_ObjSelect.py $NAME

NAME=Fencing1
python sam2_demoVideo_maskInput.py $NAME
python sam2_demoVideo_maskSingleInput.py $NAME
python sam2_demoVideoNew_maskInput.py $NAME
python sam2_demoVideoNew_maskSingleInput.py $NAME
python ComputeIOU_ObjSelect.py $NAME

NAME=Carpark1
python sam2_demoVideo_maskInput.py $NAME
python sam2_demoVideo_maskSingleInput.py $NAME
python sam2_demoVideoNew_maskInput.py $NAME
python sam2_demoVideoNew_maskSingleInput.py $NAME
python ComputeIOU_ObjSelect.py $NAME
