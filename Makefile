up:
	sudo docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
down:
	sudo docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down
	
yolo7_up:
	./serverless/deploy_gpu.sh ./serverless/onnx/WongKinYiu/yolov7/nuclio
yolo11_up:
	./serverless/deploy_gpu.sh ./serverless/onnx/yolov11

yolo7_down:
	sudo docker stop nuclio-nuclio-onnx-wongkinyiu-yolov7
yolo11_down:
	sudo docker stop nuclio-nuclio-onnx-yolov11

