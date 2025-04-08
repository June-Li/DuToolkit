# build amd64 GPU images
# 如果要构建完成后push到镜像仓库，可以执行 ./build.sh --push
set -x
CI_REGISTRY_IMAGE=hub.sensedeal.vip/library/sdai-pdfs-ocr-torch-gpu

CI_COMMIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CI_COMMIT_TAG=$(git describe --tags --exact-match 2>/dev/null)
CI_COMMIT_SHORT_SHA=$(git rev-parse --short HEAD)

VERSION=${CI_COMMIT_TAG:-$(echo $CI_COMMIT_BRANCH | sed 's/[^a-zA-Z0-9]/-/g')-$(date +%y%m%d)-$CI_COMMIT_SHORT_SHA}

docker run --privileged --rm hub.sensedeal.vip/library/multiarch/qemu-user-static:register --reset
docker run --privileged --rm hub.sensedeal.vip/library/binfmt --install all
docker buildx create --use --node base --name cubebuilder --driver-opt image=hub.sensedeal.vip/library/buildkit:buildx-stable-1
docker buildx build --platform=linux/amd64 -f dockerfiles/Dockerfile.gpu -t "$CI_REGISTRY_IMAGE:$VERSION-gpu-amd64" --load . $@
docker buildx build --platform=linux/arm64 -f dockerfiles/Dockerfile.gpu -t "$CI_REGISTRY_IMAGE:$VERSION-gpu-arm64" --load . $@
