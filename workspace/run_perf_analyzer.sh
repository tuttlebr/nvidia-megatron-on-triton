#!/bin/bash
clear

SERVER_STATUS=$(curl -m 1 -L -s -o /dev/null -w %{http_code} localhost:8000/v2/models/ensemble/versions/1/ready)

while [[ ${SERVER_STATUS} -ne 200 ]]
do 
    echo "Waiting for Triton Server to be ready..."
    SERVER_STATUS=$(curl -m 1 -L -s -o /dev/null -w %{http_code} localhost:8000/v2/models/ensemble/versions/1/ready)
    sleep 5
done

echo "Triton Server ready!"
perf_analyzer -m ensemble \
    --async \
    --percentile=99 \
    --concurrency-range 4:16:4 \
    --input-data perf_analyzer_data.json \
    -u localhost:8001 \
    -i grpc \
    --measurement-interval 30000 \
    --request-distribution poisson