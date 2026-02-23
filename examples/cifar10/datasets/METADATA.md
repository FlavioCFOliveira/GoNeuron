# CIFAR-10 Dataset Metadata

## Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Structure
The dataset is divided into five training batches (`data_batch_1.bin` to `data_batch_5.bin`) and one test batch (`test_batch.bin`), each with 10,000 images.

## Binary Format
Each file contains 10,000 records. Each record is 3073 bytes long:
- **Byte 0**: Label (0-9)
- **Bytes 1-1024**: Red channel (32x32 pixels, row-major)
- **Bytes 1025-2048**: Green channel (32x32 pixels, row-major)
- **Bytes 2049-3072**: Blue channel (32x32 pixels, row-major)

## Labels
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck

## Go Parsing Example

```go
package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

const (
	imageSize   = 32 * 32
	recordSize  = 1 + 3*imageSize // 1 byte label + 3072 bytes RGB
	numChannels = 3
)

func LoadCIFAR10(filePath string) ([][]float32, []int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	var images [][]float32
	var labels []int

	for {
		record := make([]byte, recordSize)
		_, err := io.ReadFull(reader, record)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}

		label := int(record[0])
		labels = append(labels, label)

		// Convert bytes to float32 and normalize to [0, 1]
		// The binary format is RRR...GGG...BBB...
		imgData := make([]float32, 3*imageSize)
		for i := 0; i < 3*imageSize; i++ {
			imgData[i] = float32(record[i+1]) / 255.0
		}
		images = append(images, imgData)
	}

	return images, labels, nil
}

func main() {
	images, labels, err := LoadCIFAR10("examples/cifar10/datasets/data_batch_1.bin")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d images with labels\n", len(images))
}
```

## Normalization
For GoNeuron, it is recommended to normalize the pixel values to the range [0, 1] or [-1, 1] depending on the activation function used in the input layer.
