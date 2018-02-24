data_mount="bo.nov29/datasets/squad/4"




if [ "$1" = "gpu" ]
then
echo "Using floyd $1"
floyd run --gpu --env tensorflow-1.4:py2 --data $data_mount:/data "python console.py"
elif [ "$1" = "cpu" ]
then
echo "Using floyd $1"
floyd run --env tensorflow-1.4:py2 --data $data_mount:/data "python console.py"
else
echo "Please indicate cpu or gpu"
fi
