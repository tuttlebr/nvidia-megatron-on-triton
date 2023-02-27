#!/bin/bash
clear

perf_analyzer -m ensemble \
    --async \
    --percentile=95 \
    --concurrency-range 4:16:4 \
    --input-data perf_analyzer_data.json \
    -u 127.0.0.0:8001 \
    -i grpc \
    --measurement-interval 30000