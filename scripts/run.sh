#!/usr/bin/env bash

set -e

RETRAIN=false
CONFIG_MAP=""
DATA_CONFIG=""
MODEL_CONFIG=""
EXTRAS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-map)
            CONFIG_MAP="$2"
            shift 2
            ;;
        --retrain)
            RETRAIN=true
            shift
            ;;
        --data_config)
            DATA_CONFIG="$2"
            shift 2
            ;;
        --model_config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --extras)
            EXTRAS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG_MAP" ]]; then
    echo "Missing --config-name"
    exit 1
fi

if [[ -z "$DATA_CONFIG" ]]; then
    echo "Missing --data_config"
    exit 1
fi

echo "Using config map: $CONFIG_MAP"

if [[ "$RETRAIN" == true ]]; then
    echo "Running retrain..."
    python src/train.py \
        --config-name "$CONFIG_MAP" \
        data="$DATA_CONFIG" \
        $EXTRAS
else
    if [[ -z "$MODEL_CONFIG" ]]; then
        echo "Missing --model_config"
        exit 1
    fi

    echo "Running train hook..."
    python src/train_hook.py \
        --config-name "$CONFIG_MAP" \
        data="$DATA_CONFIG" \
        model="$MODEL_CONFIG" \
        $EXTRAS
fi