# install nvidia for docker sudo apt install -y nvidia-docker2


sudo docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

sudo docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down

./serverless/deploy_cpu.sh serverless/openvino/dextr
./serverless/deploy_cpu.sh serverless/openvino/omz/public/yolo-v3-tf


sudo docker stop nuclio-nuclio-onnx-wongkinyiu-yolov7

./serverless/deploy_gpu.sh ./serverless/onnx/WongKinYiu/yolov7/nuclio
./serverless/deploy_gpu.sh ./serverless/onnx/yolov11

 nuclio    | onnx-wongkinyiu-yolov7 | cvat    | ready | 1/1      | 32768     
 nuclio    | onnx-yolov11           | cvat    | ready | 1/1      | 32769  

sudo docker stop onnx-yolov11 onnx-wongkinyiu-yolov7


watch -n 1 nuctl get functions
