up:
	sudo docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
down:
	sudo docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down

yolo_up:
	./serverless/deploy_gpu.sh ./serverless/onnx/yolov8

yolo_down:
	sudo docker stop nuclio-nuclio-onnx-yolov8

traffic_up:
	./serverless/deploy_gpu.sh ./serverless/onnx/traffic-sign
traffic_down:
	sudo docker stop nuclio-nuclio-onnx-traffic-sign


clean:
	sudo docker volume prune -a
	sudo docker image prune -a
.PHONY: up down yolo_up yolo_down traffic_up traffic_down clean