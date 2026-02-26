NAME=AlexaMeadeExhibit
python sam2_demoVideo_maskSingleInputMVSeg.py $NAME
python ComputeIOU.py $NAME

NAME=AlexaMeadeFacePaint
python sam2_demoVideo_maskSingleInputMVSeg.py $NAME
python ComputeIOU.py $NAME

NAME=CoffeeMartini
python sam2_demoVideo_maskSingleInputMVSeg.py $NAME
python ComputeIOU.py $NAME

NAME=Dog
python sam2_demoVideo_maskSingleInputMVSeg.py $NAME
python ComputeIOU.py $NAME

NAME=Frog
python sam2_demoVideo_maskSingleInputMVSeg.py $NAME
python ComputeIOU.py $NAME

NAME=Welder
python sam2_demoVideo_maskSingleInputMVSeg.py $NAME
python ComputeIOU.py $NAME
