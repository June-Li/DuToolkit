# build amd64 MLU images
# 如果要构建完成后push到镜像仓库，可以执行 ./build.sh --push
set -x
CI_REGISTRY_IMAGE=hub.sensedeal.vip/library/sdai-pdfs-ocr-torch-gpu

CI_COMMIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CI_COMMIT_TAG=$(git describe --tags --exact-match 2>/dev/null)
CI_COMMIT_SHORT_SHA=$(git rev-parse --short HEAD)

VERSION=${CI_COMMIT_TAG:-$(echo $CI_COMMIT_BRANCH | sed 's/[^a-zA-Z0-9]/-/g')-$(date +%y%m%d)-$CI_COMMIT_SHORT_SHA}

docker build -f dockerfiles/Dockerfile.mlu -t "$CI_REGISTRY_IMAGE:$VERSION-mlu-amd64" --load . $@
