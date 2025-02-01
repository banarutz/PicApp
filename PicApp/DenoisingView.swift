import SwiftUI
import CoreML
import PhotosUI

struct DenoisingView: View {
    @State private var selectedModel = "Exported Model"
    @State private var isProcessing = false
    @State private var resultImage: UIImage? = nil
    @State private var inferenceTime: TimeInterval? = nil
    @State private var inputImage: UIImage? = nil
    @State private var pickerDelegate: PickerDelegate?

    let models = ["DnCNN", "EDD"]

    var body: some View {
        VStack {
            // Dropdown for model selection
            Picker("Select Model", selection: $selectedModel) {
                ForEach(models, id: \ .self) { model in
                    Text(model).tag(model)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding()

            // Button to load image
            Button(action: loadImage) {
                Text("Load Image")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()

            // Loading indicator
            if isProcessing {
                ProgressView("Processing...")
                    .padding()
            }

            // Display the processed image
            if let resultImage = resultImage {
                Image(uiImage: resultImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 300)
                    .padding()

                // Display inference time
                if let time = inferenceTime {
                    Text("Inference Time: \(String(format: "%.2f", time)) seconds")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .padding()
                }
            } else {
                Text("Processed image will appear here.")
                    .foregroundColor(.gray)
            }

            Spacer()

            // Button to run inference
            if inputImage != nil {
                Button(action: runInference) {
                    Text("Run Denoising")
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            }
        }
        .navigationTitle("Image Denoising")
    }

    func loadImage() {
        let configuration = PHPickerConfiguration(photoLibrary: .shared())
        let picker = PHPickerViewController(configuration: configuration)

        let delegate = PickerDelegate { image in
            DispatchQueue.main.async {
                self.inputImage = image
            }
        }

        self.pickerDelegate = delegate
        picker.delegate = delegate

        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootViewController = windowScene.windows.first?.rootViewController {
            rootViewController.present(picker, animated: true)
        }
    }

    func runInference() {
        guard let inputImage = inputImage else { return }
        isProcessing = true
        resultImage = nil
        inferenceTime = nil

        DispatchQueue.global(qos: .userInitiated).async {
            let startTime = CFAbsoluteTimeGetCurrent()

            let modelName: String
            switch selectedModel {
            case "DnCNN":
                modelName = "DnCNN_SIDD_small_50x50_experiment_1_exported"
            case "EDD":
                modelName = "DnCNN_SIDD_small_50x50_experiment_1_traced"
            default:
                DispatchQueue.main.async {
                    print("Invalid model selected")
                    isProcessing = false
                }
                return
            }

            guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc"),
                  let model = try? MLModel(contentsOf: modelURL) else {
                DispatchQueue.main.async {
                    print("Failed to load or compile model: \(modelName)")
                    isProcessing = false
                }
                return
            }

            guard let patches = preprocessImage(inputImage) else {
                DispatchQueue.main.async {
                    print("Failed to preprocess image")
                    isProcessing = false
                }
                return
            }

            var processedPatches: [UIImage] = []
            for patch in patches {
                if let pixelBuffer = patch.toCVPixelBuffer() {
                    let inputKey = "input"
                    let outputKey = "output"
                    if let input = try? MLDictionaryFeatureProvider(dictionary: [inputKey: pixelBuffer]),
                       let prediction = try? model.prediction(from: input),
                       let outputBuffer = prediction.featureValue(for: outputKey)?.imageBufferValue,
                       let processedImage = UIImage(pixelBuffer: outputBuffer) {
                        processedPatches.append(processedImage)
                    }
                }
            }

            let reconstructedImage = reconstructImage(from: processedPatches, originalSize: inputImage.size)

            let endTime = CFAbsoluteTimeGetCurrent()
            DispatchQueue.main.async {
                resultImage = reconstructedImage
                inferenceTime = endTime - startTime
                isProcessing = false
            }
        }
    }

    func preprocessImage(_ image: UIImage) -> [UIImage]? {
        guard let cgImage = image.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        let patchSize = 50

        var patches: [UIImage] = []
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        for y in stride(from: 0, to: height, by: patchSize) {
            for x in stride(from: 0, to: width, by: patchSize) {
                let context = CGContext(
                    data: nil,
                    width: patchSize,
                    height: patchSize,
                    bitsPerComponent: 8,
                    bytesPerRow: patchSize * 4,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                )

                context?.setFillColor(UIColor.black.cgColor)
                context?.fill(CGRect(x: 0, y: 0, width: patchSize, height: patchSize))
                context?.draw(cgImage, in: CGRect(x: -x, y: -y, width: width, height: height))

                if let patchImage = context?.makeImage() {
                    patches.append(UIImage(cgImage: patchImage))
                }
            }
        }

        return patches
    }

    func reconstructImage(from patches: [UIImage], originalSize: CGSize) -> UIImage? {
        let width = Int(originalSize.width)
        let height = Int(originalSize.height)
        let patchSize = 50
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )

        var index = 0
        for y in stride(from: 0, to: height, by: patchSize) {
            for x in stride(from: 0, to: width, by: patchSize) {
                if index < patches.count, let cgImage = patches[index].cgImage {
                    context?.draw(cgImage, in: CGRect(x: x, y: y, width: patchSize, height: patchSize))
                    index += 1
                }
            }
        }

        if let finalImage = context?.makeImage() {
            return UIImage(cgImage: finalImage)
        }

        return nil
    }
}

class PickerDelegate: NSObject, PHPickerViewControllerDelegate {
    let completion: (UIImage?) -> Void

    init(completion: @escaping (UIImage?) -> Void) {
        self.completion = completion
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let result = results.first else {
            completion(nil)
            return
        }

        if result.itemProvider.canLoadObject(ofClass: UIImage.self) {
            result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] object, _ in
                self?.completion(object as? UIImage)
            }
        } else {
            completion(nil)
        }
    }
}

extension UIImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            return nil
        }

        UIGraphicsBeginImageContext(CGSize(width: width, height: height))
        defer { UIGraphicsEndImageContext() }

        guard let cgImage = self.cgImage else {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        return pixelBuffer
    }
}

extension UIImage {
    convenience init?(pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)

        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }

        self.init(cgImage: cgImage)
    }
}
