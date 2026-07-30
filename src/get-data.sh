#!/bin/sh
# Downloads the MNIST dataset into data/
set -e

BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"
DATA_DIR="$(dirname "$0")/data"

fetch() {
    src_name="$1"
    dst_name="$2"
    size="$3"

    dst="$DATA_DIR/$dst_name"
    if [ -f "$dst" ] && [ $(wc -c < "$dst") -eq "$size" ]; then
        echo "$dst_name already present, skipping"
        return
    fi

    echo "Downloading $src_name.gz ..."
    curl -fSL "$BASE_URL/$src_name.gz" -o "$dst.gz"
    gzip -df "$dst.gz"

    if [ $(wc -c < "$dst") -ne "$size" ]; then
        echo "error: $dst_name is $(wc -c < "$dst") bytes, expected $size" >&2
        exit 1
    fi
}

mkdir -p "$DATA_DIR"
fetch train-images-idx3-ubyte train-images.idx3-ubyte 47040016
fetch train-labels-idx1-ubyte train-labels.idx1-ubyte 60008
fetch t10k-images-idx3-ubyte t10k-images.idx3-ubyte  7840016
fetch t10k-labels-idx1-ubyte t10k-labels.idx1-ubyte  10008
echo "MNIST data ready in $DATA_DIR"
