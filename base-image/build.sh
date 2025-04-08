sudo docker buildx create --use --node base --name cubebuilder --driver-opt image=hub.sensedeal.vip/library/buildkit:buildx-stable-1
sudo docker buildx build --platform linux/arm64,linux/amd64  -t hub.sensedeal.vip/library/sdai-ocr-base:20240712 --push . 
