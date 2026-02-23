# ML Architecture Expert Memory

## Patterns and Conventions
- **Layout Consistency**: This project uses NCHW (Channel-major) layout for 2D layers (`Conv2D`, `MaxPool2D`, `BatchNorm2D`).
- **Activation Functions**: `LogSoftmax` should be used with `NLLLoss` for classification. `Dense` layer explicitly handles `LogSoftmax` and `Softmax` for gradient computation.
- **Hardware Acceleration**: Metal support is preferred on Darwin/arm64. Use `layer.NewMetalDevice()` and `network.SetDevice(metal)`.

## Implementation Insights
- **BatchNorm2D**: Updated from NHWC to NCHW to match `Conv2D` output. Indexing is now `idx := f*numel + s` where `f` is feature/channel and `s` is spatial position.
- **CIFAR-10 Data**: 3073 bytes per record (1 label + 3072 RGB). Normalize pixels to [0, 1] by dividing by 255.
- **Zero Allocation**: Always reuse `outputBuf`, `preActBuf`, and other buffers in layer implementations.

## Debugging Notes
- If `LogSoftmax` is used, ensure the subsequent layer or loss handles it (e.g., `Dense.Backward` or `NLLLoss`). `LogSoftmax.Derivative` panics by design.
