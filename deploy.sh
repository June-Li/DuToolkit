#!/bin/bash

if [ -z "$1" ]; then
    env="test"
else
    env="$1"
fi

if [ "$env" = "prod" ]; then
    curl -X PATCH \
          -H "content-type: application/strategic-merge-patch+json" \
          -H "Authorization:Bearer $K8S_TOKEN" \
          -d "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"cubeflow-cubefile\",\"image\":\"$CI_REGISTRY_IMAGE:$VERSION\"}]}}}}" \
          "http://kuboard.sensedeal.wiki/k8s-api/apis/apps/v1/namespaces/cubechat/deployments/cubeflow-cubefile"
elif [ "$env" = "test" ]; then
    curl -X PUT \
        -H "content-type: application/json" \
        -H "Cookie: KuboardUsername=gongxiangfeng; KuboardAccessKey=$K8S_TOKEN_TEST" \
        -d "{\"kind\":\"deployments\",\"namespace\":\"cubechat\",\"name\":\"cubefile-deployment\",\"images\":{\"$CI_REGISTRY_IMAGE\":\"$CI_REGISTRY_IMAGE:$VERSION\"}}" \
        "http://kuboard.test.sensedeal.wiki:50080/kuboard-api/cluster/k8s-test/kind/CICDApi/gongxiangfeng/resource/updateImageTag"
fi
